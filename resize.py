import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from multiprocessing.pool import Pool



def Resize(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #for gray-scale images
    image = cv2.resize(image,(512, 512))
    return image

def Resize_images(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        imagemat = Resize(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt+=1
    print("\n\n--saved in " + destdir + "--\n")

sourcedir = ('/home/linh/Downloads/data/ori/BIRAD_4C')
destdir = ('/home/linh/Downloads/data/ori_resized/BIRAD_4C')
os.makedirs(destdir, exist_ok=True)
print("The new directory is created!")
with Pool(28) as p:
    p.map(Resize_images(sourcedir, destdir))
