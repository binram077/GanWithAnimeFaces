import tensorflow as tf
import keras
import keras.layers as layers
import GAN
from matplotlib import pyplot as plt
import numpy as np

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "images", label_mode=None, image_size=(64, 64), batch_size=128
)
dataset = dataset.map(lambda x: x / 255.0)

g = GAN.Gan(64)
g.compile(
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss_fn=tf.losses.binary_crossentropy,
)
g.load()
g.fit(
    dataset,batch_size=32, epochs=1,callbacks=[GAN.GANMonitor(5,64)]
)
g.save()

"""generated_ims = g.generator(latent_vector)
for i in range(32):
  plt.imshow(np.squeeze(generated_ims[i]),cmap = 'gray')
  plt.show()
g.generator.summary()"""

"""
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "images", label_mode=None, image_size=(64, 64), batch_size=128
)
dataset = dataset.map(lambda x: x / 255.0)
for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break

gan = Gan(64)
gan.compile(
    d_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=tf.optimizers.Adam(learning_rate=0.0001),
    loss_fn=tf.losses.binary_crossentropy,
)
gan.fit(
    dataset,batch_size=32, epochs=20,callbacks=[GANMonitor(5,64)]
)
gan.save()
latent_vector = np.random.normal(size = (32,gan.latent_dim))
generated_ims = gan.generator(latent_vector)
for i in range(32):
  plt.imshow(np.squeeze(generated_ims[i]),cmap = 'gray')
  plt.show()"""