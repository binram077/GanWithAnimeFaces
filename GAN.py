import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np
from keras import layers
print(tf.executing_eagerly())

from numba import cuda
device = cuda.get_current_device()
device.reset()

class Gan(keras.Model):
    def __init__(self, latent_dim,names = ['new']):
        super().__init__()
        if names[0] == 'new':
            self.discriminator = self.get_Discriminator()
            self.generator = self.get_Generator(latent_dim)
        elif len(names) == 2:
            self.discriminator = keras.models.load_model(names[0])
            self.generator = keras.models.load_model(names[1])
        else:
            raise "Problem"
        self.latent_dim = latent_dim

    def get_Discriminator(self):
        d = keras.Sequential(
            [
                keras.Input(shape=(64, 64, 3)),
                layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Flatten(),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        return d

    def get_Generator(self, latent_dim):
        g = keras.Sequential(
            [
                keras.Input(shape=(latent_dim,)),
                layers.Dense(8 * 8 * 32),
                layers.Reshape((8, 8, 32)),
                layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(64, kernel_size=5, activation="sigmoid", padding="same"),
                layers.LeakyReLU(alpha=0.2),
                layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid"),
            ]
        )
        return g

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    @tf.function
    def train_step(self, r_images):
        d_f_images = self.generator(tf.random.normal([tf.shape(r_images)[0],self.latent_dim]))
        d_r_labels = tf.ones([tf.shape(d_f_images)[0], 1])
        d_f_labels = tf.zeros([tf.shape(r_images)[0],1])
        with tf.GradientTape() as d_tape:
            d_r_preds = self.discriminator(r_images)
            d_f_preds = self.discriminator(d_f_images)
            d_r_loss = self.loss_fn(d_r_labels, d_r_preds)
            d_f_loss = self.loss_fn(d_f_labels, d_f_preds)
            d_total_loss = d_r_loss + d_f_loss
        d_grads = d_tape.gradient(d_total_loss,self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads,self.discriminator.trainable_weights))

        g_f_labels = tf.ones([tf.shape(d_f_images)[0], 1])
        latent_vector = tf.random.normal([tf.shape(r_images)[0],self.latent_dim])
        with tf.GradientTape() as g_tape:
            g_f_images = self.generator(latent_vector)
            g_f_preds = self.discriminator(g_f_images)
            g_loss = self.loss_fn(g_f_labels, g_f_preds)
        g_grads = g_tape.gradient(g_loss,self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_total_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }

    def save(self,model_names = ['disc.h','gen.h'],opt_names = ["d_opt.npy","g_opt.npy"]):
        self.discriminator.save(model_names[0])
        self.generator.save(model_names[1])
        np.save(opt_names[0],self.d_optimizer.get_weights())
        np.save(opt_names[1],self.g_optimizer.get_weights())

    def load(self,model_names = ['disc.h','gen.h'],opt_names = ["d_opt.npy","g_opt.npy"]):
        self.discriminator = keras.models.load_model(model_names[0])
        self.generator = keras.models.load_model(model_names[1])
        #we need to use the optimizer once so it will know the size of weights it should get
        #so we will aplly zero grads on both discriminator and generator
        zero_grads_disc = [tf.zeros_like(w) for w in self.discriminator.trainable_weights]
        zero_grads_gen = [tf.zeros_like(w) for w in self.generator.trainable_weights]
        self.d_optimizer.apply_gradients(zip(zero_grads_disc,self.discriminator.trainable_weights))
        self.g_optimizer.apply_gradients(zip(zero_grads_gen, self.generator.trainable_weights))
        self.d_optimizer.set_weights(np.load(opt_names[0],allow_pickle=True))
        self.g_optimizer.set_weights(np.load(opt_names[1],allow_pickle=True))

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            print("gotcha")
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("ims\\generated_img_%02d_%d.png" % (epoch, i))
