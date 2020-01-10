from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np 
import cv2 as cv 
import tensorflow as tf 

#import tensorflow_datasets as tfds
# https://www.tensorflow.org/datasets/catalog/cycle_gan

# https://www.tensorflow.org/tutorials/generative/cyclegan
# https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
# https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628

import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import dataio 

import llayers 
from llayers import upsample, downsample
from tensorflow_examples.models.pix2pix import pix2pix

import subpixel

tf.summary.trace_on(
    graph=True,
    profiler=False
)

#tfds.disable_progress_bar()
AUTOTUNE = tf.data.experimental.AUTOTUNE
cuneiform_dataset_directory = '/home/alex/data/cdli/image_translation/dataset'

BUFFER_SIZE = 1000
BATCH_SIZE = 32
NUM_VIS = BATCH_SIZE
IMG_WIDTH = 256
IMG_HEIGHT = 256
#IMG_HEIGHT,IMG_WIDTH = int(640./4), int(480./4)
EPOCHS = 3000
LAMBDA = 1 # default 10

LABEL_SMOOTH_HIGH = 0.9
NORM_TYPE = ['batchnorm','instancenorm'][0]

def discriminator(norm_type='batchnorm'):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  
  implementation taken from https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py#L318
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='input_image')
  x = inp

  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

  #return tf.keras.Model(inputs=inp, outputs=[down1,down2,down3,zero_pad2, last])
  return tf.keras.Model(inputs=inp, outputs=last)

loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gradient_x(img):
  return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):
  return img[:, :-1, :, :] - img[:, 1:, :, :]

def depth_smoothness(depth, img):
  """Computes image-aware depth smoothness loss."""
  depth_dx = gradient_x(depth)
  depth_dy = gradient_y(depth)
  #print('depth gradient',depth_dx.get_shape().as_list(),depth_dy.get_shape().as_list())
  image_dx = gradient_x(img)
  image_dy = gradient_y(img)
  #print('image gradient',image_dx.get_shape().as_list(),image_dy.get_shape().as_list())
  weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_dx), 3, keepdims=True))
  weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_dy), 3, keepdims=True))
  #print('weights',weights_x.get_shape().as_list(),weights_y.get_shape().as_list())
  smoothness_x = depth_dx * weights_x
  smoothness_y = depth_dy * weights_y
  #print('smoothness xy',smoothness_x.get_shape().as_list(),smoothness_y.get_shape().as_list())
  sm = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(abs(smoothness_y))
  return sm

def calc_ssim_loss(x, y):
  """Computes a differentiable structured image similarity measure."""
  c1 = 0.01**2
  c2 = 0.03**2
  mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
  mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')
  sigma_x = tf.nn.avg_pool2d(x**2, 3, 1, 'VALID') - mu_x**2
  sigma_y = tf.nn.avg_pool2d(y**2, 3, 1, 'VALID') - mu_y**2
  sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y
  ssim_n = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
  ssim_d = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
  ssim = ssim_n / ssim_d
  ssim = tf.clip_by_value((1 - ssim) / 2, 0, 1)
  ssim = tf.reduce_mean(ssim)
  return ssim

def discriminator_loss(real, generated):
  real_loss = loss_obj(LABEL_SMOOTH_HIGH * tf.ones_like(real), real)

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

def generate_images(prediction,test_input,step,vis_directory):
  fn = os.path.join(vis_directory,'cuneiform-cyclegan-%i.png' % step)
  #print('test input',test_input.shape,prediction.shape)    
  #plt.figure(figsize=(NUM_VIS * 10, 12))

  images = [np.hstack([test_input[i,:,:,:], prediction[i,:,:,:]]) for i in range(NUM_VIS)]

  count_per_row = 4

  mosaic = np.zeros((1,1+count_per_row* images[0].shape[1],3),'uint8')
  mosaic_row = np.zeros((images[0].shape[0],1,3),'uint8')
  
  for i,image in enumerate(images):
    mosaic_row = np.hstack((mosaic_row,image))
    if i % count_per_row == count_per_row-1:
      mosaic = np.vstack((mosaic,mosaic_row))
      mosaic_row = np.zeros((images[0].shape[0],1,3),'uint8')

  mosaic_uint8 = np.uint8(np.around(255*(mosaic - mosaic.min())/(mosaic.max()-mosaic.min())))

  cv.imwrite(fn,mosaic_uint8)
  return True 
  

    
def train(data,config):
  
  dataset = dataio.load_cuneiform_dataset(cuneiform_dataset_directory,BATCH_SIZE)

  train_horses, train_zebras = dataset['trainA'], dataset['trainB']
  test_horses, test_zebras = dataset['testA'], dataset['testB']
  bs= BATCH_SIZE

  sample_horse = next(iter(train_horses))
  sample_zebra = next(iter(train_zebras))

  sample_horses = [next(iter(train_horses)) for _ in range(NUM_VIS)]

  OUTPUT_CHANNELS = 3

  generator_g = llayers.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)
  generator_f = llayers.unet_generator(OUTPUT_CHANNELS, norm_type=NORM_TYPE)

  discriminator_x = llayers.discriminator(norm_type=NORM_TYPE,target=False)
  discriminator_y = llayers.discriminator(norm_type=NORM_TYPE,target=False)

  generator_g_optimizer = tf.keras.optimizers.Adam(config['lr']['G'], beta_1=0.5)
  generator_f_optimizer = tf.keras.optimizers.Adam(config['lr']['F'], beta_1=0.5)

  discriminator_x_optimizer = tf.keras.optimizers.Adam(config['lr']['Dx'], beta_1=0.5)
  discriminator_y_optimizer = tf.keras.optimizers.Adam(config['lr']['Dy'], beta_1=0.5)

  # checkpoints
  # checkpoints and tensorboard summary writer
  now = str(datetime.now()).replace(' ','_').replace(':','-')
  checkpoint_path = "./checkpoints/cyclegan_cuneiform/%s" % now
  vis_directory = os.path.join(checkpoint_path,'vis')
  if not os.path.isdir(vis_directory):
    os.makedirs(vis_directory)

  writer = tf.summary.create_file_writer(checkpoint_path)

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
  def gaussian_noise_layer(input_layer, std):
      noise = tf.random.normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
      return input_layer + noise

  #@tf.function
  def train_step(real_x, real_y, writer, global_step, should_summarize = False):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
      # Generator G translates X -> Y
      # Generator F translates Y -> X.
      
      fake_y = generator_g(real_x, training=True)
      fake_x = generator_f(real_y, training=True)
      
      # blur fake symbols to get rid of high frequency noise
      if 0:
        blur_size = 3
        blur_kernel = tf.ones([blur_size,blur_size,3,3],tf.float32)/(blur_size * blur_size)
        fake_y = tf.nn.conv2d(fake_y,blur_kernel,strides=1,padding='SAME')
        fake_x = tf.nn.conv2d(fake_x,blur_kernel,strides=1,padding='SAME')
    
      cycled_x = generator_f(fake_y, training=True)
      cycled_y = generator_g(fake_x, training=True)

      disc_real_x = discriminator_x(gaussian_noise_layer(real_x, .1), training=True)
      disc_real_y = discriminator_y(gaussian_noise_layer(real_y, .1), training=True)

      disc_fake_x = discriminator_x(gaussian_noise_layer(fake_x, .1), training=True)
      disc_fake_y = discriminator_y(gaussian_noise_layer(fake_y, .1), training=True)

      # calculate the loss
      gen_g_loss = generator_loss(disc_fake_y)
      gen_f_loss = generator_loss(disc_fake_x)
      
      
      #depth_smoothness_loss = depth_smoothness(fake_y,real_y)

      # cycle loss
      cycle_loss_x, cycle_loss_y = calc_cycle_loss(real_x, cycled_x), calc_cycle_loss(real_y, cycled_y)
      
      # ssim
      ssim_loss_x = calc_ssim_loss(real_x, cycled_x) #*2.
      ssim_loss_y = calc_ssim_loss(real_y, cycled_y) #*2.
      total_cycle_loss = cycle_loss_x + cycle_loss_y + ssim_loss_x + ssim_loss_y
      
      # Total generator loss = adversarial loss + cycle loss
      total_gen_g_loss = gen_g_loss + total_cycle_loss# + depth_smoothness_loss#
      total_gen_f_loss = gen_f_loss + total_cycle_loss#  

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

    # write summary
    if should_summarize:
      with tf.device("cpu:0"):
        with writer.as_default():
          tf.summary.scalar("total loss generator G",total_gen_g_loss,step=global_step)
          tf.summary.scalar("total loss generator F",total_gen_f_loss,step=global_step)
          tf.summary.scalar("loss generator G",gen_g_loss,step=global_step)
          tf.summary.scalar("loss generator F",gen_f_loss,step=global_step)
          tf.summary.scalar("loss cycle x",cycle_loss_x,step=global_step)
          tf.summary.scalar("loss cycle y",cycle_loss_y,step=global_step)
          tf.summary.scalar("loss ssim x",ssim_loss_x, step= global_step)
          tf.summary.scalar("loss ssim y",ssim_loss_y, step=global_step)
          tf.summary.scalar("loss discrim x",disc_x_loss,step=global_step)
          tf.summary.scalar("loss discrim y",disc_y_loss,step=global_step)
          #tf.summary.scalar("loss depth_smoothness", depth_smoothness_loss, step=global_step)

          def im_summary(name,data):
            tf.summary.image(name,(data+1)/2,step=global_step)
          
          im_summary("x",real_x)
          im_summary("G(x)",fake_y)
          im_summary("F(G(x))",cycled_x)
          im_summary("y",real_y)
          im_summary("F(y)",fake_x)
          im_summary("G(F(y))",cycled_y)
          #tf.summary.image("discrim x",disc,step=global_step)
          
        
        writer.flush()    

  n = 0
  for epoch in range(EPOCHS):
    start = time.time()
    
    for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
      if n % 25 == 0:
        generate_images(generator_g(sample_horses), sample_horse,n,vis_directory)
      _global_step = tf.convert_to_tensor(n, dtype=tf.int64)
      try:
        train_step(image_x, image_y, writer, _global_step, should_summarize = n%20==0)
      except Exception as e:
        print()
        print('error on step',n,e)
        print()
      
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
    lr = 1e-4 # default 2e-4
    config = {
      'lr': {"G":lr/4.,"F":lr/4.,"Dx":lr,"Dy":lr}
    }
    train(data,config)
    print()
    #vis_sample(data[0][0],data[0][1])