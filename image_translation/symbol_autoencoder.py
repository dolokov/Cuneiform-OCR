"""

    Denoising Convolutional Autoencoder on single drawn symbols
    hidden representation is later used for unsupervised cluster finding
"""

import os
import numpy as np 
import tensorflow as tf 
from glob import glob 
from random import shuffle 
import time 

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

import llayers
from datetime import datetime

single_symbol_dataset_directory = '/data/cdli/symbols/patches'
BATCH_SIZE = 16
n_latentcode = 64

def load_image_train_single_symbol(image_file):
  # load from disk
  image = tf.io.read_file(image_file)
  image = tf.image.decode_png(image)
  image = tf.cast(image, tf.float32)

  # resize to 
  image = tf.image.resize(image,(64,64))

  # norm [-1,1]
  image = (image / 127.5) - 1
  if image.get_shape().as_list()[2] == 3:
    image = image[:,:,0]
    #image = tf.tile(image,[1,1,3])  
  return image

def Encoder(config,inputs):
    x = inputs
    # denoising
    #x = tf.keras.layers.GaussianNoise(0.2)(x)
    
    x = llayers.downsample_stridedconv(32,(3,3), norm_type='batchnorm', apply_norm=False)(x) # 32
    x = llayers.downsample_stridedconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 16
    x = llayers.downsample_stridedconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 8
    x = llayers.downsample_stridedconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 4
    x = llayers.downsample_stridedconv(256,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 2
    x = llayers.downsample_stridedconv(n_latentcode,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 1
    return x

def Decoder(config,encoder):
    x = encoder
    x = llayers.upsample_transpconv(n_latentcode,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 2
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 4
    x = llayers.upsample_transpconv(128,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 8
    x = llayers.upsample_transpconv(64,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 16
    x = llayers.upsample_transpconv(32,(3,3), norm_type='batchnorm', apply_norm=True)(x) # 32
    x = llayers.upsample_transpconv(1,(3,3), norm_type='batchnorm', apply_norm=False,activation=tf.tanh)(x) # 64
    return x 

def load_single_symbols_dataset(dataset_dir,config):
    source_files = glob(os.path.join(dataset_dir,'*.png'))
    shuffle(source_files)
    #source_files = source_files[:100]
    print('[*] loaded %i symbol images' % len(source_files))
    ratio = .9999
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
    
    data = tf.data.Dataset.from_tensor_slices(data['train'])
    data = data.batch(config['batch_size'])
    return files['train'], data  

def train(config):
    symbol_files, dataset = load_single_symbols_dataset(single_symbol_dataset_directory,config)

    inputs = tf.keras.layers.Input(shape=[config['img_height'], config['img_width'], 1])
    encoder = Encoder(config,inputs)
    reconstructed = Decoder(config,encoder)

    optimizer = tf.keras.optimizers.Adam(config['lr'])
    encoder_model = Model(inputs = inputs, outputs = encoder)
    autoencoder = Model(inputs = inputs, outputs = reconstructed) #dataset['train'][0],outputsdataset['train'][1])

    # checkpoints and tensorboard summary writer
    now = str(datetime.now()).replace(' ','_').replace(':','-')
    checkpoint_path = os.path.expanduser("~/checkpoints/autoencoder_singlesymbol/%s/model" % now)
    vis_directory = os.path.join(checkpoint_path,'vis')
    logdir = os.path.join(checkpoint_path,'logs')
    for _directory in [checkpoint_path,logdir]:
        if not os.path.isdir(_directory):
            os.makedirs(_directory)

    writer = tf.summary.create_file_writer(checkpoint_path)
    def show_tb_images(batch_step):
        with tf.device("cpu:0"):
            with writer.as_default():    
                _global_step = tf.convert_to_tensor(batch_step, dtype=tf.int64)
                tf.summary.image("input",(inputs+1.)/2.,step=batch_step)
                tf.summary.image("reconstructed",(reconstructed+1)/2,step=batch_step)
      
    ckpt = tf.train.Checkpoint(encoder=encoder_model,
                            autoencoder=autoencoder,
                            optimizer = optimizer)
    
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('[*] Latest checkpoint restored',ckpt_manager.latest_checkpoint)

    def train_step(inp, writer, global_step, should_summarize = False):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # autoencode inputs
            reconstructed_inp = autoencoder(inp, training=True)
            
            # L1 loss
            loss = tf.reduce_mean(tf.abs(reconstructed_inp - inp))

        # gradients
        gradients = tape.gradient(loss,autoencoder.trainable_variables)
        # update weights
        optimizer.apply_gradients(zip(gradients,autoencoder.trainable_variables))

        # write summary
        if should_summarize:
            with tf.device("cpu:0"):
                with writer.as_default():
                    tf.summary.scalar("l1 loss",loss,step=global_step)
                    def im_summary(name,data):
                        tf.summary.image(name,(data+1)/2,step=global_step)
                    im_summary('image',inp)
                    im_summary('reconstructed',reconstructed_inp)
                    writer.flush()    
    n = 0
    for epoch in range(config['epochs']):
        start = time.time()

        for inp in dataset:
            _global_step = tf.convert_to_tensor(n, dtype=tf.int64)

            train_step(inp,writer,_global_step,should_summarize=n%20==0)
            n+=1
        
        end = time.time()
        print('[*] epoch %i took %f seconds.'%(epoch,end-start))

    ckpt_save_path = ckpt_manager.save()
    print('Saving checkpoint for epoch {} at {}'.format(config['epochs'], ckpt_save_path))

    """ after training the autoencoder, encode complete dataset and save to disk """
    file_encodings = 'single_symbol_encodings.csv'
    with open(file_encodings,'w') as f:
        for i,inp in enumerate(dataset):
            features = encoder_model(inp,training=False)
            features = np.array(features)
            features = features.reshape([-1,n_latentcode])
            for b in range(features.shape[0]):
                symbol_file = symbol_files[features.shape[0]*i+b]    
                line = '%s,%s'%(symbol_file,','.join([str(xx) for xx in features[b]]))
                f.write(line+'\n')
    return file_encodings

def save_encoding():
    config = {'batch_size':BATCH_SIZE, 'img_height':64,'img_width':64}
    config['epochs'] = 100
    config['lr'] = 1e-3

    file_encodings = train(config)
    print('[*] saved encodings to %s' % file_encodings)

if __name__ == '__main__':
    save_encoding()