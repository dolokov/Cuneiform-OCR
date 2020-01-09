from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import cv2 as cv 
import tensorflow as tf 

#import tensorflow_datasets as tfds
# https://www.tensorflow.org/datasets/catalog/cycle_gan

# https://www.tensorflow.org/tutorials/generative/cyclegan


import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

import llayers 
from llayers import upsample, downsample, Discriminator
from tensorflow_examples.models.pix2pix import pix2pix


#tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE

BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
#IMG_HEIGHT,IMG_WIDTH = int(640./4), int(480./4)
EPOCHS = 300
LAMBDA = 10.
def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH,3])#, image.get_shape().as_list()[2]])

  return cropped_image

# normalizing the images to [-1, 1] 
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  if image.get_shape().as_list()[2] == 1:
    image = tf.tile(image,[1,1,3])
  image = tf.reshape(image,(IMG_HEIGHT,IMG_WIDTH,3))
  print('shape',image.get_shape().as_list())
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [int(1.5*IMG_HEIGHT),int(1.5*IMG_WIDTH)],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image

def preprocess_image_train(image, label=None):
  image = normalize(image)
  image = random_jitter(image)
  return image

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)

  image = tf.cast(image, tf.float32)
  #print('image',image.get_shape())
  return image

def load_image_train(image_file):
  input_image = load(image_file)
  input_image = normalize(input_image)
  input_image = random_jitter(input_image)
  print('input',input_image.shape)
  return input_image

def load_dataset(path):
  train_dataset = tf.data.Dataset.list_files(path)
  train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.batch(BATCH_SIZE)
  return train_dataset

def vis_sample(train_horses,train_zebras):
    sample_horse = next(iter(train_horses))
    sample_zebra = next(iter(train_zebras))

    plt.subplot(121)
    plt.title('Horse')
    plt.imshow(sample_horse[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Horse with random jitter')
    plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)

    plt.subplot(121)
    plt.title('Zebra')
    plt.imshow(sample_zebra[0] * 0.5 + 0.5)

    plt.subplot(122)
    plt.title('Zebra with random jitter')
    plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)






generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1


def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

def generate_images(model, test_input,step):
  prediction = model(test_input)
    
  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  if 0:
    plt.show()
  else:
    fn = '/tmp/horsezebra-%i.png' % step
    plt.savefig(fn)
    print('[*] saved %s'%fn)
    
def train(data,config):
  import datetime
  log_dir="logs/"

  import tensorflow_datasets as tfds
  dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

  train_horses, train_zebras = dataset['trainA'], dataset['trainB']
  test_horses, test_zebras = dataset['testA'], dataset['testB']
  bs= BATCH_SIZE

  train_horses = train_horses.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
      BUFFER_SIZE).batch(bs)

  train_zebras = train_zebras.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
      BUFFER_SIZE).batch(bs)

  test_horses = test_horses.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
      BUFFER_SIZE).batch(bs)

  test_zebras = test_zebras.map(
      preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
      BUFFER_SIZE).batch(bs)

  sample_horse = next(iter(train_horses))
  sample_zebra = next(iter(train_zebras))

  OUTPUT_CHANNELS = 3

  generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
  generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

  discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
  discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

  generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
  discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

  # checkpoints
  checkpoint_path = "./checkpoints/cyclegan_horsezebra"

  ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            generator_f=generator_f,
                            discriminator_x=discriminator_x,
                            discriminator_y=discriminator_y,
                            generator_g_optimizer=generator_g_optimizer,
                            generator_f_optimizer=generator_f_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            discriminator_y_optimizer=discriminator_y_optimizer)

  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')



  @tf.function
  def train_step(real_x, real_y):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.
      
      fake_y = generator_g(real_x, training=True)
      cycled_x = generator_f(fake_y, training=True)

      fake_x = generator_f(real_y, training=True)
      cycled_y = generator_g(fake_x, training=True)

      # same_x and same_y are used for identity loss.
      same_x = generator_f(real_x, training=True)
      same_y = generator_g(real_y, training=True)

      disc_real_x = discriminator_x(real_x, training=True)
      disc_real_y = discriminator_y(real_y, training=True)

      disc_fake_x = discriminator_x(fake_x, training=True)
      disc_fake_y = discriminator_y(fake_y, training=True)

      # calculate the loss
      gen_g_loss = generator_loss(disc_fake_y)
      gen_f_loss = generator_loss(disc_fake_x)
      
      total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
      
      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
      total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

      disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
      disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    
    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                          generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                          generator_f.trainable_variables)
    
    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                              discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                              discriminator_y.trainable_variables)
    
    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                              generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                              generator_f.trainable_variables))
    
    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                  discriminator_x.trainable_variables))
    
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                  discriminator_y.trainable_variables))


  n = 0
  for epoch in range(EPOCHS):
    start = time.time()
    
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
      if n % 100 == 0:
        clear_output(wait=True)
        # Using a consistent image (sample_horse) so that the progress of the model
        # is clearly visible.
        generate_images(generator_g, sample_horse,n)

      train_step(image_x, image_y)
      if n % 10 == 0:
        print ('.', end='')
      n+=1


    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                          ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))

if __name__ == '__main__':
    #data  = load_datasets()
    #print(data)
    data= None
    config = {}
    train(data,config)
    print()
    #vis_sample(data[0][0],data[0][1])