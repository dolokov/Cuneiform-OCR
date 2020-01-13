"""
    take images of different resolutions and create fixed size images by padding black
    make 256x256 patches
"""
import os
import numpy as np 
import cv2 as cv 
from glob import glob 

import extract_symbol_patches
import photo2gray

directory_images = '/home/alex/data/cdli/images'
directory_source = '/home/alex/data/cdli/image_translation/dataset/source'
directory_target = '/home/alex/data/cdli/image_translation/dataset/target'

for _dir in [directory_source,directory_target]:
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

def create_photo_dataset(res=256):
    files = glob(os.path.join(directory_images,'*.png'))
    if 0:
        shapes = []
        for f in files:
            shapes.append(cv.imread(f).shape[:2])
        shapes = np.array(shapes)
        print('mean',shapes.mean(axis=0))
        print('min',shapes.min(axis=0))
        print('max',shapes.max(axis=0))
    
    count = 0
    for i,f in enumerate(files):
        im = cv.imread(f)
        #im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        #im = cv.cvtColor(im,cv.COLOR_GRAY2BGR)
        im = photo2gray.preprocess(im)

        #if len(im.shape)==3 and im.shape[2]==3:
        #    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
        pad = (im.shape[0]-im.shape[1])//2
        if im.shape[0] > im.shape[1]: # higher than wider, pad left and right
            sstack = np.hstack
        else:
            sstack = np.vstack

        im = sstack((np.zeros((im.shape[0],pad),'uint8'),im,np.zeros((im.shape[0],pad),'uint8')))
        k = 20
        #for loop_res in [(res,res),(res-k,res+k),(res+k,res-k),(res,res-k),(res,res+k),(res-k,res),(res+k,res),(res-k,res-k)]:
        for loop_res in [(res,res)]:#
            resized = cv.resize( im, loop_res )
            
            # random h/v flips
            """for do_hflip in [False,True]:
                for do_vflip in [False,True]:
                    ima = np.uint8(resized) 
                    if do_vflip:
                        ima = cv.flip(ima,0)
                    if do_hflip:
                        ima = cv.flip(ima,1)"""
            fno = os.path.join(directory_source,'%i.png' % count)
            cv.imwrite(fno,np.uint8(resized) )
            count += 1
    return count 
    
def create_symbols_dataset(num_samples):
    url = 'https://cdli.ucla.edu/dl/lineart/P429857_ld.jpg'
    fn_in = os.path.expanduser('~/P429857_ld.jpg')
    if not os.path.isfile(fn_in):
        import subprocess
        subprocess.call(['wget',str(url)])
        if not os.path.isfile(fn_in):
            raise Exception('[*** ERROR ***] please download image from %s to local file %s'% (url,fn_in) )

    if not os.path.isdir(directory_target):
        os.makedirs(directory_target)
    directory_patches_symbols = '/tmp/%i'%int(np.random.uniform(1e6))
    if not os.path.isdir(directory_patches_symbols):
        os.makedirs(directory_patches_symbols)

    symbols = extract_symbol_patches.extract_symbols(fn_in, directory_patches_symbols)
    extract_symbol_patches.make_cuniform_symbols(symbols, target_dir = directory_target,num_samples = num_samples)

if __name__ == '__main__':
    print('[*] creating source photo dataset ...')
    num_samples = create_photo_dataset()
    print('[*] creating target symbol dataset ...')
    create_symbols_dataset(num_samples)