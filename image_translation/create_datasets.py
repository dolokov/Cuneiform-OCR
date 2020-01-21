"""
    take images of different resolutions and create fixed size images by padding black
    make 256x256 patches
"""
import os
import numpy as np 
import cv2 as cv 
from glob import glob 
import pickle
from pathlib import Path

import extract_symbol_patches
import photo2gray

directory_images = os.path.expanduser('~/data/cdli/images')
directory_source = os.path.expanduser('~/data/cdli/image_translation/dataset/source')
directory_target = os.path.expanduser('~/data/cdli/image_translation/dataset/target')

pkl_files = {'train': os.path.expanduser('~/data/cdli/object_detection/train.pkl')}

for _dir in [directory_images,directory_source,directory_target]:
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

def create_photo_dataset(res=256):
    files = glob(os.path.join(directory_images,'*.png'))
    if len(files) == 0 :
        print("""
            found no images in %s! :(

                please download and extract them from
                https://drive.google.com/open?id=1cJlr8AxH0i1FikxBZRt17yhf3TsRROLu
        """)
        assert 1==0
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
    
def create_symbols_dataset_single(num_samples):
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

# write a function that loads the dataset into detectron2's standard format
def load_boundingboxes(mode='train'):
    """ https://gilberttanner.com/blog/train-a-microcontroller-detector-using-detectron2 """
    
    bounding_boxes = pickle.load( open(pkl_files['train'],'rb'))

    dataset_dicts = []
    for filename in bounding_boxes.keys():
        record = {}
        
        record["file_name"] = filename
        record["height"] = bounding_boxes[filename][0]['height']
        record["width"] = bounding_boxes[filename][0]['width']

        objs = []
        for d in bounding_boxes[filename]:
          obj= {
              'bbox': d['bbox'],
              'bbox_mode': BoxMode.XYXY_ABS,
              'category_id': d['class_id'],
              "iscrowd": 0
          }
          objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


def create_symbols_dataset(num_samples, directory_target = '~/data/cdli/symbols/cuniforms'):
    directory_target = os.path.expanduser(directory_target)
    target_dir = directory_target
    if not os.path.isdir(directory_target):
        os.makedirs(directory_target)
    
    clusters = {}
    ## load clusters
    cluster_base = os.path.expanduser('~/data/cdli/symbol_clusters/eps_3.000000')
    cluster_directories = glob(os.path.join(cluster_base,'*'))
    for x in cluster_directories:
        clusters[x.split('/')[-1]] = [cv.imread(fn) for fn in glob(os.path.join(x,'*.png'))]
    n_clusters = len(clusters.keys())

    def make_cuniform_symbols(clusters, res = 256 ):
        """
            take single symbols and combine them to kind of realistic cuniform tablet transcriptions

            has random symbol size per image 
            has random padding per line of symbols
            has random number of lines of symbols (3-8)
            each line of symbols contains 3 - 10 symbols
            each line of symbols globally drawn with 95%
            each symbol in line random with 70% drawn
            separation lines drawn globally with certain proba per image(92.5%)
            each separation line drawn with certain proba per line
        """

        symbol_size = (48/2,48/2)
        line_color = 0
        H,W = int(640./4), int(480./4)
        #print('H/W',H,'/',W)
        for j,(cluster_dir,cluster) in enumerate(clusters.items()):
            for i, symbol in enumerate(cluster):
                symbscale = symbol_size[0] / np.max(symbol.shape[:2])
                symbol = cv.resize(symbol,(0,0),fx=symbscale,fy=symbscale)
                #if symbol.shape[-1]==3:
                #    symbol = cv.cvtColor(symbol,cv.COLOR_BGR2GRAY)
                clusters[cluster_dir][i] = symbol 
        
        bounding_boxes = {}
        for count_samples in range(num_samples):
            im = 255 * np.ones((H,W,3),'uint8')
            padd = int(np.random.uniform(5,15))
            num_lines_of_symbols = int(np.random.uniform(3,9))
            should_draw_horizontal_lines = np.random.uniform() < 0.925
            file_boxes = []
            for count_line in range(num_lines_of_symbols):
                y = int(H*1.*count_line/num_lines_of_symbols)
                num_line_symbols = int(np.random.uniform(4,11))
                # draw symbols of line
                for count_symbol in range(num_line_symbols):
                    x = int(count_symbol * symbol_size[1] + padd)
                    # symbol = symbols[int(np.random.uniform(len(symbols)))] # random symbol
                    id_cluster = list(clusters.keys())[int(np.random.uniform(n_clusters))]
                    symbol = clusters[id_cluster][int(np.random.uniform(len(clusters[id_cluster])))]

                    # randomly downscale symbol
                    if np.random.uniform() < 0.5:
                        new_size = np.array(symbol.shape[:2])
                        new_size = np.int32(np.around(np.random.uniform(0.25,1.0,size=(2,)) * new_size))
                        symbol = cv.resize(symbol,tuple(new_size))

                    #print('symbolshape',symbol.shape)
                    if np.random.uniform() < 0.70:
                        try:#if x<im.shape[1]-symbol.shape[1]:
                            im[y:y+symbol.shape[0],x:x+symbol.shape[1]] = symbol

                            # save bounding box
                            h,w = symbol.shape[:2]
                            file_boxes.append({'class':int(id_cluster),'bbox':[x,y,x+w,y+h],'width':im.shape[1],'height':im.shape[0]})
                        except Exception as e:
                            pass
                            #print(e)

                # draw horizontal line random
                if should_draw_horizontal_lines and np.random.uniform() < .9: 
                    cv.line(im,(0,y),(W-1,y),line_color,2)

            # padd to square
            #if len(im.shape)==3 and im.shape[2]==3:
            #    im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
            pad = (im.shape[0]-im.shape[1])//2
            if im.shape[0] > im.shape[1]: # higher than wider, pad left and right
                sstack = np.hstack
            else:
                sstack = np.vstack

            im = sstack((np.zeros((im.shape[0],pad,3),'uint8'),im,np.zeros((im.shape[0],pad,3),'uint8')))
            # scale to resolution
            im = cv.resize( im, (res,res) )


            fn = os.path.join(target_dir,'%i.png'%count_samples)
            cv.imwrite(fn, im)
            #print('wrote',fn)

            if not fn in bounding_boxes:
                bounding_boxes[fn] = []

            bounding_boxes[fn].extend(file_boxes)

        # write_boxes
        pickle.dump(bounding_boxes, open(pkl_files['train'],'wb'))


        return bounding_boxes

    bounding_boxes = make_cuniform_symbols(clusters)

    return clusters 

if __name__ == '__main__':
    print('[*] creating source photo dataset ...')
    num_samples = create_photo_dataset()
    print('[*] creating target symbol dataset ...')
    create_symbols_dataset(num_samples)