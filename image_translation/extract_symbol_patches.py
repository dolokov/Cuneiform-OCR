
import numpy as np
import cv2 as cv
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors


def get_random_color():
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    return colors[int(np.random.uniform(len(colors)))]


def extract_symbols(filepath, patch_dir):
    im = cv.imread(filepath)
    print(im.shape)

    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    (thresh, bw) = cv.threshold(gray, 128,
                                255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    bw = 255 - bw
    kernel = np.uint8(np.ones((3, 3)))
    dilation = cv.dilate(bw, kernel, iterations=3)
    #cv.imwrite('/tmp/dilated.png', dilation)

    xxxx = cv.findContours(
        dilation.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    try:
        contours, hierarchy = xxxx
    except:
        _, contours, hierarchy = xxxx

    # cut 5% highest resolution contours
    contours = sorted(contours, key=cv.contourArea)[:int(0.95*len(contours))]

    if not os.path.isdir(patch_dir):
        os.makedirs(patch_dir)

    min_dim = 20
    vis = 255 * np.ones_like(im)
    count_patches = 0
    shapes = []
    symbols = []
    for cnt in contours:
        cc = np.reshape(cnt, (-1, 2))
        mins, maxs = np.min(cc, axis=0), np.max(cc, axis=0)
        patch = im[mins[1]:maxs[1], mins[0]:maxs[0]]
        if patch is not None and np.min(patch.shape[:2]) > min_dim:
            filepath = os.path.join(patch_dir, '%i.png' % count_patches)
            count_patches += 1
            if count_patches % 100 == 0:
                ''#print(count_patches, patch.shape)
            #cv.imwrite(filepath, patch)
            symbols.append(patch)
            shapes.append(patch.shape[:2])

            #vis = cv.rectangle(
            #    vis, (mins[0], mins[1]), (maxs[0], maxs[1]), get_random_color(), 2)
    #vis = cv.drawContours(vis,[cnt],0,color,-1)
    #cv.imwrite('/tmp/patches.png', vis)
    shapes = np.array(shapes)
    print('[*] symbol shapes:')
    print('mean/std x', np.mean(shapes[:, 1]), '/', np.std(shapes[:, 1]))
    print('mean/std y', np.mean(shapes[:, 0]), '/', np.std(shapes[:, 0]))
    print('min/max x', np.min(shapes[:, 1]), '/', np.max(shapes[:, 1]))
    print('min/max y', np.min(shapes[:, 0]), '/', np.max(shapes[:, 0]))
    """
        mean/std x 29.621932256101683 / 5.47480473894983
        mean/std y 26.118788452437293 / 1.57455703246771
        min/max x 21 / 66
        min/max y 21 / 72
    """

    return symbols 

def make_cuniform_symbols(symbols, num_samples = 2000, res = 256, target_dir = '/data/cdli/symbols/cuniforms' ):
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

    if not os.path.isdir(target_dir):os.makedirs(target_dir)
    symbol_size = (48/2,48/2)
    line_color = 0
    H,W = int(640./4), int(480./4)
    #print('H/W',H,'/',W)
    for i,symbol in enumerate(symbols):
        symbscale = symbol_size[0] / np.max(symbol.shape[:2])
        symbol = cv.resize(symbol,(0,0),fx=symbscale,fy=symbscale)
        #if symbol.shape[-1]==3:
        #    symbol = cv.cvtColor(symbol,cv.COLOR_BGR2GRAY)
        symbols[i] = symbol 
        
    for count_samples in range(num_samples):
        im = 255 * np.ones((H,W,3),'uint8')
        padd = int(np.random.uniform(5,15))
        num_lines_of_symbols = int(np.random.uniform(3,9))
        should_draw_horizontal_lines = np.random.uniform() < 0.925
        for count_line in range(num_lines_of_symbols):
            y = int(H*1.*count_line/num_lines_of_symbols)
            num_line_symbols = int(np.random.uniform(4,11))
            # draw symbols of line
            for count_symbol in range(num_line_symbols):
                x = int(count_symbol * symbol_size[1] + padd)
                symbol = symbols[int(np.random.uniform(len(symbols)))]
                # randomly downscale symbol
                if np.random.uniform() < 0.5:
                    new_size = np.array(symbol.shape[:2])
                    new_size = np.int32(np.around(np.random.uniform(0.25,1.0,size=(2,)) * new_size))
                    symbol = cv.resize(symbol,tuple(new_size))

                #print('symbolshape',symbol.shape)
                if np.random.uniform() < 0.70:
                    try:#if x<im.shape[1]-symbol.shape[1]:
                        im[y:y+symbol.shape[0],x:x+symbol.shape[1]] = symbol
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


def get_data(patch_dim=48, patch_dir='/data/cdli/symbols/patches'):
    from glob import glob
    files = glob(os.path.join(patch_dir, '*.png'))
    imgs = [cv.imread(f) for f in files]


if __name__ == '__main__':
    fn_in = "/data/cdli/P429857_ld.jpg"
    patch_dir = '/data/cdli/symbols/patches'
    symbols = extract_symbols(fn_in, patch_dir)
    make_cuniform_symbols(symbols)