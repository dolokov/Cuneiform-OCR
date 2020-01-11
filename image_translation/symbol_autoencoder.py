"""

    Denoising Convolutional Autoencoder on single drawn symbols
    hidden representation is later used for unsupervised cluster finding
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import llayers
from datetime import datetime

single_symbol_dataset_directory = '/data/cdli/symbols/patches'
BATCH_SIZE = 16

def load_image_train_single_symbol(image_file):
  # load from disk
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)
  image = tf.cast(image, tf.float32)

  # resize to 
  image = tf.image.resize(image,(64,64))

  # norm
  image = (image / 127.5) - 1
  if image.get_shape().as_list()[2] == 3:
    image = image[:,:,0]
    #image = tf.tile(image,[1,1,3])
  
  return image

def Encoder(config,inputs):
    x = inputs
    x = llayers.downsample_stridedconv(32,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 32
    x = llayers.downsample_stridedconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 16
    x = llayers.downsample_stridedconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 8
    x = llayers.downsample_stridedconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 4
    x = llayers.downsample_stridedconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 2
    x = llayers.downsample_stridedconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 1
    #return tf.keras.Model(inputs=inputs, outputs=x)
    return x

def Decoder(config,encoder):
    #inputs = tf.keras.layers.Input(shape=[1,1, 256],tensor=encoder.outputs[0])
    #x = inputs
    x = encoder
    x = llayers.upsample_transpconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 32
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 16
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 8
    x = llayers.upsample_transpconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 4
    x = llayers.upsample_transpconv(32,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 2
    x = llayers.upsample_transpconv(1,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 1
    #return tf.keras.Model(inputs=inputs, outputs=x)
    return x 

def load_single_symbols_dataset(dataset_dir,config):
    source_files = glob(os.path.join(dataset_dir,'*.png'))
    shuffle(source_files)
    #source_files = source_files[:100]
    print('[*] loaded %i symbol images' % len(source_files))
    ratio = .9
    files={'train':source_files[:int(len(source_files)*ratio)],'test':source_files[int(len(source_files)*ratio):]}
    data = {}
    import cv2 as cv 
    for mode in ['train','test']:
        data[mode] = []
        for f in files[mode]:
            im = cv.imread(f)
            im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            im = cv.resize(im,(64,64))
            im = im.astype(np.float32)
            im = (im / 127.5) - 1
            im = np.expand_dims(im,axis=3)
            data[mode].append(im)
        data[mode] = np.array(data[mode])
    return data  

    for ratio,mode in zip([0.9,0.1],['train','test']):
        BUFFER_SIZE = 1000
        data[mode] = tf.data.Dataset.list_files(files[mode])
        data[mode] = data[mode].map(load_image_train_single_symbol, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    
        data[mode] = data[mode].shuffle(BUFFER_SIZE)
        
        # add noisy variant to make it supervised
        noise_factor = 0.5
        @tf.function
        def get_noisy(unnoisy):
            print('unnoise',unnoisy)
            noise = unnoisy
            noise = unnoisy + noise_factor * tf.random.normal([config['img_height'],config['img_width'],3], mean=0.0, stddev=1.0)#unnoisy.get_shape().as_list()
            #noise = noise[:,:,0]
            #noise = tf.expand_dims(noise,axis=2)
            noise = tf.clip_by_value(noise, -1., 1.)
            return noise 
        
        data[mode] = data[mode].batch(config['batch_size'])
        data[mode] = data[mode].prefetch(4*config['batch_size'])#.cache()
        
        #data[mode] = [data[mode],data[mode].map(get_noisy,num_parallel_calls=tf.data.experimental.AUTOTUNE)]
        #data[mode] = np.array([xx for xx in data[mode]])
        data[mode] = np.array(data[mode])
        print('huhu data',data[mode].shape)
    return data 



def train(config):
    dataset = load_single_symbols_dataset(single_symbol_dataset_directory,config)

    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 1])
    encoder = Encoder(config,inputs)
    decoder = Decoder(config,encoder)

    optimizer = tf.keras.optimizers.Adam(config['lr'])

    #encoded = encoder(training=True)
    #reconstructed = decoder(training=True)
    encoder_model = Model(inputs = inputs, outputs = encoder)
    autoencoder = Model(inputs = inputs, outputs = decoder) #dataset['train'][0],outputsdataset['train'][1])
    autoencoder.compile(optimizer=optimizer, loss='mae')

    # checkpoints
    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-')
    checkpoint_path = os.path.join("~/checkpoints/autoencoder_singlesymbol/%s" % now)
    vis_directory = os.path.join(checkpoint_path,'vis')
    logdir = os.path.join(checkpoint_path,'logs')
    for _directory in [checkpoint_path,logdir]:
        if not os.path.isdir(_directory):
            os.makedirs(_directory)

    writer = tf.summary.create_file_writer(checkpoint_path)

    """ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)"""
    ckpt = tf.train.Checkpoint(encoder=encoder_model,
                            autoencoder=autoencoder,
                            optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,update_freq=50)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path)
    autoencoder.fit(dataset['train'],dataset['train'],
        batch_size=config['batch_size'],
        epochs=config['epochs'],
        shuffle=True,
        validation_data=(dataset['test'],dataset['test']),
        callbacks=[tensorboard_callback,checkpoint_callback],
    )
    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(config['epochs'], ckpt_save_path))

if __name__ == '__main__':
    config = {'batch_size':BATCH_SIZE, 'img_height':64,'img_width':64}
    config['epochs'] = 50
    config['lr'] = 1e-4

    train(config)