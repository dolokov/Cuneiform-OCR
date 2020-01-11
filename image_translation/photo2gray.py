import os 
import numpy as np 
import cv2 as cv 
from glob import glob 


directory_rgb = '/data/cdli/cuniforms/images'
directory_vis = '/tmp/gray_photos'
if not os.path.isdir(directory_vis): os.makedirs(directory_vis)

def preprocess(rgb):
    '''hsv = cv.cvtColor(rgb,cv.COLOR_BGR2HSV)
    h,s,v = cv.split(hsv) # v is almost identical to gray image
    hsvc = np.hstack((v,h,s))
    return hsvc'''
    
    gray = cv.cvtColor(rgb,cv.COLOR_BGR2GRAY)
    
    alpha = 1.5 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)

    #gray = cv.convertScaleAbs(gray, alpha=alpha, beta=beta)
    sigmaColor,sigmaSpace=75,75
    sigmaColor,sigmaSpace=15,15
    ks = 9
    ks = 5
    gray  = cv.bilateralFilter(gray,ks,sigmaColor,sigmaSpace)

    # create a CLAHE object (Arguments are optional).
    tileGridSize=(8,8)
    tileGridSize=(2,2)
    
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize)
    contrast_improved = clahe.apply(gray)

    return contrast_improved

def main():
    files = sorted(glob(os.path.join(directory_rgb,'*.png')))
    for i,f in enumerate(files):
        im = cv.imread(f)
        print(i,im.shape,f)

        gray = cv.cvtColor(im,cv.COLOR_BGR2GRAY)

        preprocessed = preprocess(im)

        vis = np.hstack((im,cv.cvtColor(gray,cv.COLOR_GRAY2BGR),cv.cvtColor(preprocessed,cv.COLOR_GRAY2BGR)))
        fno = os.path.join(directory_vis,f.split('/')[-1])
        cv.imwrite(fno,vis)
        

if __name__ == '__main__':
    main()