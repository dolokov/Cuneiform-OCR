
import os,sys
import numpy as np 
import cv2 as cv 

def extract_images():

    filepath = "/home/dolokov/Cuneiform-OCR/object-detection/Mask_RCNN/annotation/annotation.csv"
    image_directory = '/data/cdli/cuniforms/images'

    with open(filepath,'r') as f:
        lines = f.readlines()
        boxes = {}
        for i, line in enumerate(lines):
            line = line.replace('\n','')

            #print(i,line)
            parts = line.split(',')

            resolution = [int(q) for q in parts[0].split('x')]
            print()
            print('line res',resolution)
            print(line)
            filename = parts[1]

            class_type = int(parts[2])

            bbox_str = parts[3]
            #print(bbox_str)

            bbox_strings = [p for p in parts if ':' in p and '-' in p]
            print(i)
            #for bb in bbox_strings:
            #    print('->',bb)

            """
            pparts = line.split('_')
            pparts = pparts[1:]
            for pp in pparts:
                if '.png:' in pp:
                    print('-->',pp.split(':'))
                    good_parts = pp.split(':')
                    fnim = os.path.join(image_directory,good_parts[0])
                    im = cv.imread(fnim)
                    im = cv.resize(im,tuple(resolution))
                    x,y,w,h = [cc for cc in good_parts[1:5]]   
                    print('H crop',h,good_parts[0],im.shape)        
                    hsplit = h.split('-')
                    if len(hsplit) == 2:   
                        h,classtype = hsplit
                    x,y,w,h = [int(cc) for cc in [x,y,w,h]]
                    #patch = im[y:y+h,x:x+w]
                    patch = im[x:x+w][y:y+h]
                    if patch is not None and np.min(patch.shape)>0:
                        cv.imwrite('/tmp/bb-%i.png'%int(np.random.uniform(1e5)),patch)
            """

if __name__ == '__main__':
    extract_images()