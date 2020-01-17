## https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

import tensorflow as tf 

use_spectral_norm = True 
if use_spectral_norm:
  import spectral_norm
  from spectral_norm import Conv2D
else:
  from tf.keras.layers import Conv2D

class GrayLayer(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super(GrayLayer, self).__init__(kwargs)

  def call(self,inputs):
    x = inputs
    if 1:
      #print('xA',x.get_shape().as_list())
      x = tf.reduce_mean(x,axis=3)
      x = tf.expand_dims(x,axis=3)
      x = tf.tile(x,[1,1,1,3])
      #print('xB',x.get_shape().as_list())
      return x 
    else:
      # attention module 
      r = x[:,:,:,0]
      g = x[:,:,:,1]
      b = x[:,:,:,2]

      x = r*g+b 
      x = tf.expand_dims(x,axis=3)
      x = tf.tile(x,[1,1,1,3])
      
      return x

def downsample_stridedconv(filters, size, norm_type='batchnorm', apply_norm=True):
  """Downsamples an input.
  Conv2D => Batchnorm => LeakyRelu
  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_norm: If True, adds the batchnorm layer
  Returns:
    Downsample Sequential Model
  """
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

class DownSampling2D(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super(DownSampling2D, self).__init__(kwargs)
    
  def call(self, inputs):
    _,h,w,_ = inputs.get_shape().as_list()
    return tf.image.resize(inputs,(int(w/2),int(h/2)),method=tf.image.ResizeMethod.BILINEAR,antialias=True)

def downsample_nn(filters, size, norm_type='batchnorm', apply_norm=True):
  

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  
  result.add(DownSampling2D())
  result.add(
      Conv2D(filters, size, strides=1, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result



def upsample_subpixel(filters, size, norm_type='batchnorm', apply_norm=True, apply_dropout=True):
  import subpixel

  result = tf.keras.Sequential()
  result.add(subpixel.Subpixel(filters, size,2))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())


  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.BatchNormalization())    

  result.add(tf.keras.layers.ReLU())

  return result



def upsample_transpconv(filters, size, norm_type='batchnorm', apply_norm=True, apply_dropout=False,activation=tf.nn.relu):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())


  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.Activation(activation))
  
  #result.add(tf.keras.layers.ReLU())

  return result

def upsample_nn(filters, size, norm_type='batchnorm', apply_norm=True, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()

  result.add(tf.keras.layers.UpSampling2D(interpolation=['nearest','bilinear'][1]))

  result.add(
    Conv2D(filters, size, strides=1,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  if apply_norm:
    if norm_type.lower() == 'batchnorm':
      result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
      result.add(InstanceNormalization())


  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result 

downsample = [downsample_stridedconv,downsample_nn][0]
upsample = [upsample_transpconv,upsample_subpixel, upsample_nn][2]

def unet_generator(output_channels, norm_type='batchnorm', img_height=256,img_width=256):
  """Modified u-net generator model (https://arxiv.org/abs/1611.07004).
  Args:
    output_channels: Output channels
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
  Returns:
    Generator model
  """
  print('unet_generator')
  initializer = tf.random_normal_initializer(0., 0.02)
  ks = 4
  last = tf.keras.layers.Conv2DTranspose(
      output_channels, ks, strides=2,
      padding='same', kernel_initializer=initializer,
      activation='tanh')  # (bs, 256, 256, 3)

  concat = tf.keras.layers.Concatenate()

  inputs = tf.keras.layers.Input(shape=[img_height, img_width, output_channels])
  x = inputs

  # Downsampling through the model
  ss = 1
  
  down_stack = [
      downsample(64, ks, norm_type, apply_norm=False),  # (bs, 128, 128, 64)
      downsample(128, ks, norm_type),  # (bs, 64, 64, 128)
      downsample(256, ks, norm_type),  # (bs, 32, 32, 256)
      downsample(512, ks, norm_type),  # (bs, 16, 16, 512)
      downsample(ss*512, ks, norm_type),  # (bs, 8, 8, 512)
      #downsample(ss*512, ks, norm_type),  # (bs, 4, 4, 512)
      #downsample(ss*512, ks, norm_type),  # (bs, 2, 2, 512)
      #downsample(ss*512, ks, norm_type),  # (bs, 1, 1, 512)
  ]

  up_stack = [
      #upsample_transpconv(ss*512, ks, norm_type, apply_dropout=True),  # (bs, 2, 2, 1024)
      #upsample_transpconv(ss*512, ks, norm_type, apply_dropout=True),  # (bs, 4, 4, 1024)
      #upsample_transpconv(ss*512, ks, norm_type, apply_dropout=True),  # (bs, 8, 8, 1024)
      upsample(512, ks, norm_type),  # (bs, 16, 16, 1024)
      upsample(256, ks, norm_type),  # (bs, 32, 32, 512)
      upsample(128, ks, norm_type),  # (bs, 64, 64, 256)
      upsample(64, ks, norm_type),  # (bs, 128, 128, 128)
  ]
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for i,(up, skip) in enumerate(zip(up_stack, skips)):
    x = up(x)
    x = concat([x, skip])
  
  intermediate = x 

  x = last(x)

  # RGB->Gray->RGB
  #x = GrayLayer()(x)
  
  return tf.keras.Model(inputs=inputs, outputs=x)


def discriminator(norm_type='batchnorm', target=False, img_width=256,img_height=256,output_channels=1):
  """PatchGan discriminator model (https://arxiv.org/abs/1611.07004).
  Args:
    norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    target: Bool, indicating whether target image is an input or not.
  Returns:
    Discriminator model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[img_width, img_height, output_channels], name='input_image')
  x = inp
  
  down1 = downsample(64, 4, norm_type, False)(x)  # (bs, 128, 128, 64)
  down2 = downsample(128, 4, norm_type)(down1)  # (bs, 64, 64, 128)
  down3 = downsample(256, 4, norm_type)(down2)  # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
  conv = Conv2D(
      512, 4, strides=1, kernel_initializer=initializer,
      use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

  if norm_type.lower() == 'batchnorm':
    norm1 = tf.keras.layers.BatchNormalization()(conv)
  elif norm_type.lower() == 'instancenorm':
    norm1 = InstanceNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(norm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

  last = Conv2D(
      1, 4, strides=1,
      kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)
  return tf.keras.Model(inputs=inp, outputs=[last,leaky_relu])
