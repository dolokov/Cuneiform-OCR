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


single_symbol_dataset_directory = '/home/alex/data/cdli/image_translation/dataset/target'
BATCH_SIZE = 16

def load_image_train_single_symbol(image_file):
  # load from disk
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)
  image = tf.cast(image, tf.float32)

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
    return tf.keras.Model(inputs=inputs, outputs=x)

def Decoder(config,encoded):
    #inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 1])
    x = encoded
    x = llayers.upsample_transpconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 32
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 16
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 8
    x = llayers.upsample_transpconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 4
    x = llayers.upsample_transpconv(32,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 2
    x = llayers.upsample_transpconv(1,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 1
    return tf.keras.Model(inputs=inputs, outputs=x)


def load_single_symbols_dataset(dataset_dir,config):
    source_files = glob(os.path.join(dataset_dir,'*.png'))
    shuffle(source_files)
    ratio = .9
    files={'train':source_files[:int(len(source_files)*ratio)],'test':source_files[int(len(source_files)*ratio):]}
    for ratio,mode in zip([0.9,0.1],['train','test']):
        BUFFER_SIZE = 4000
        data[mode] = tf.data.Dataset.list_files(files[mode])
        data[mode] = data[mode].map(load_image_train_single_symbol, num_parallel_calls = tf.data.experimental.AUTOTUNE).shuffle(BUFFER_SIZE).batch(config['batch_size']).prefetch(4*bs)#.cache()
    
        # add noisy variant to make it supervised
        noise_factor = 0.5
        def get_noisy(unnoisy):
            noise = unnoisy + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=unnoisy.shape)
            noise = np.clip(nosie, 0., 1.)
            return [noise,unnoisy] 
        data[mode] = data[mode].map(get_noisy,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        
    return data 



def train(config):
    dataset = load_single_symbols_dataset(single_symbol_dataset_directory,config['batch_size'])

    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 1])
    encoder = Encoder(config,inputs)
    decoder = Decoder(config,encoder)

    optimizer = tf.keras.optimizers.Adam(config['lr'])

    encoded = encoder(inputs,training=True)
    reconstructed = decoder(encoded.layers[-1], training=True)(encoded)
    autoencoder = Model(inputs = inputs, outputs = reconstructed) #dataset['train'][0],outputsdataset['train'][1])
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # checkpoints
    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-')
    checkpoint_path = "./checkpoints/cyclegan_cuneiform/%s" % now
    vis_directory = os.path.join(checkpoint_path,'vis')
    if not os.path.isdir(vis_directory):
        os.makedirs(vis_directory)

    writer = tf.summary.create_file_writer(checkpoint_path)

    ckpt = tf.train.Checkpoint(encoder=encoder,
                            decoder=decoder,
                            optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)


    autoencoder.fit(dataset['train'][0], dataset['train'][1],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        shuffle=True,
        validation_data=(x_test_noisy, x_test),
    )
    

if __name__ == '__main__':
    config = {'batch_size':BATCH_SIZE, 'img_height':64,'img_width':64}
    config['lr'] = 1e-4

    train(config)