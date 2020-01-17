import tensorflow as tf 
from glob import glob 
from random import shuffle 
import os 
import cv2 as cv 

IMG_HEIGHT = 256
IMG_WIDTH = 256

dtype = [tf.bfloat16,tf.float32][1]

def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH])#, image.get_shape().as_list()[2]])

  return cropped_image

# normalizing the images to [-1, 1] 
def normalize(image,output_channels=1):
  image = tf.cast(image, dtype)
  image = (image / 127.5) - 1
  #if output_channels==3 and image.get_shape().as_list()[2] == 1:
  #  image = tf.tile(image,[1,1,3])
  
  '''try:
    image = tf.reshape(image,(IMG_HEIGHT,IMG_WIDTH,3))
    print('before grayscal',image.get_shape().as_list())
    image = tf.image.rgb_to_grayscale(image)
  except:'''
  #image = tf.image.rgb_to_grayscale(image)
  image = image[:,:,0]
  image = tf.reshape(image,(IMG_HEIGHT,IMG_WIDTH,1))
    
  #if output_channels == 1 and image.get_shape().as_list()[2] == 3:
  #if output_channels == 1:
    
    
  
  
  #print('shape',image.get_shape().as_list())
  print('normalize')
  return image

def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [int(1.1*IMG_HEIGHT),int(1.1*IMG_WIDTH)],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  #image = tf.image.random_flip_left_right(image)

  return image

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)

  image = tf.cast(image, dtype)
  #print('image',image.get_shape())
  return image

def load_image_train(image_file):
  input_image = load(image_file)
  input_image = tf.image.resize(input_image,(256,256))
  input_image = normalize(input_image)
  print('before resize')
  
  
  #input_image = random_jitter(input_image)
  print('input image',input_image.shape)

  #input_image = tf.image.random_flip_left_right(input_image)
  #input_image = tf.image.random_flip_up_down(input_image)
  

  return input_image

def load_cuneiform_dataset(dataset_dir,bs):
    source_files = glob(os.path.join(dataset_dir,'source/*.png'))
    target_files = glob(os.path.join(dataset_dir,'target/*.png'))
    data = {'trainA':[],'testA':[],'trainB':[],'testB':[]}
    for i,_files in enumerate([source_files,target_files]):
        n = ['A','B'][i]
        k = ['source','target'][i]
        shuffle(_files)
    
        #ims = [cv.imread(f) for f in _files]
        cc = int(len(_files)*.90)
        """data['train%s'%n] = tf.data.Dataset.from_tensors(ims[:cc])
        data['test%s'%n] = tf.data.Dataset.from_tensors(ims[cc:])

        data['train%s'%n] = data['train%s'%n].cache().shuffle(1000).batch(bs)
        data['test%s'%n] = data['test%s'%n].cache().shuffle(1000).batch(bs)
        """
        # train_source = train_source.map(
        #  preprocess_image_train, num_parallel_calls=AUTOTUNE).cache().shuffle(
        #BUFFER_SIZE).batch(bs)
        BUFFER_SIZE = 1000
        data['train%s'%n] = tf.data.Dataset.list_files(os.path.join(dataset_dir,k,'*.png'))
        data['train%s'%n] = data['train%s'%n].map(load_image_train, num_parallel_calls = tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(bs).prefetch(4*bs)#.cache()
        data['test%s'%n] = []
    return data 
  