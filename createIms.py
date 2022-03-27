import tensorflow as tf
import keras
import keras.layers as layers
import GAN
from matplotlib import pyplot as plt
import numpy as np

gen = keras.models.load_model('gen.h')

random_latent_vectors = tf.random.normal(shape=(200, 64))
generated_images = gen(random_latent_vectors)
generated_images *= 255
generated_images.numpy()
for i in range(200):
    print("gotcha")
    img = keras.preprocessing.image.array_to_img(generated_images[i])
    img.save("endIms\\generated_img_%d.png" % (i))