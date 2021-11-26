import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
from multiprocessing.pool import Pool


#applying filter on a single image
def Prewitt_v1(filename, filter):
    print("reading file---> " + str(filename))
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) #for gray-scale images
    img = cv2.resize(img,(512, 512))
    #comment out the above line if there is memory issue i.e. need to resize all images to smaller dim
    h, w = img.shape # height and width of images
    print("shape: height " + str(h)+" x width " + str(w) + "\n")

    # define filters
    horizontal = filter
    vertical   = np.transpose(filter)

    # define images with 0s
    newhorizontalImage = np.zeros((h, w))
    newverticalImage   = np.zeros((h, w))
    newgradientImage   = np.zeros((h, w))

    # offset by 1
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            horizontalGrad = (horizontal[0, 0] * img[i - 1, j - 1]) + \
                             (horizontal[0, 1] * img[i - 1, j]) + \
                             (horizontal[0, 2] * img[i - 1, j + 1]) + \
                             (horizontal[1, 0] * img[i, j - 1]) + \
                             (horizontal[1, 1] * img[i, j]) + \
                             (horizontal[1, 2] * img[i, j + 1]) + \
                             (horizontal[2, 0] * img[i + 1, j - 1]) + \
                             (horizontal[2, 1] * img[i + 1, j]) + \
                             (horizontal[2, 2] * img[i + 1, j + 1])

            newhorizontalImage[i - 1, j - 1] = abs(horizontalGrad)

            verticalGrad = (vertical[0, 0] * img[i - 1, j - 1]) + \
                           (vertical[0, 1] * img[i - 1, j]) + \
                           (vertical[0, 2] * img[i - 1, j + 1]) + \
                           (vertical[1, 0] * img[i, j - 1]) + \
                           (vertical[1, 1] * img[i, j]) + \
                           (vertical[1, 2] * img[i, j + 1]) + \
                           (vertical[2, 0] * img[i + 1, j - 1]) + \
                           (vertical[2, 1] * img[i + 1, j]) + \
                           (vertical[2, 2] * img[i + 1, j + 1])

            newverticalImage[i - 1, j - 1] = abs(verticalGrad)

            # Edge Magnitude
            mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
            newgradientImage[i - 1, j - 1] = mag

    return newgradientImage


def Prewitt_v2(image):
       print("reading file---> " + str(image))
       image = cv2.imread(image, cv2.IMREAD_GRAYSCALE) #for gray-scale images
       image = cv2.resize(image, (512, 512))
      # Prewitt operator
       kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=int)
       kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=int)
       x = cv2.filter2D(image, cv2.CV_16S, kernelx)
       y = cv2.filter2D(image, cv2.CV_16S, kernely)

       # Turn uint8, image fusion
       absX = cv2.convertScaleAbs(x)
       absY = cv2.convertScaleAbs(y)
       Prewitt_2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
       return Prewitt_v2

#function for creating all edge-images of a directory
def converter_Prewitt_v1(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):

        #applying Prewitt filter
        #for appyling any other filter change filter value accordingly i.e. the 2nd args for Prewitt filter()
        #imagemat = Prewitt_v1(filename, np.array([[-1,0,1], [-1,0,1], [-1,0,1]]))
        imagemat = Prewitt_v2(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


#function for creating all edge-images of a directory
def converter_Prewitt_v1(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        #applying Prewitt filter
        #for appyling any other filter change filter value accordingly i.e. the 2nd args for Prewitt filter version 1
        imagemat = Prewitt_v1(filename, np.array([[-1,0,1], [-1,0,1], [-1,0,1]]))
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")


#function for creating all edge-images of a directory
def converter_Prewitt_v2(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        #applying Prewitt filter
        #for appyling any other filter change filter value accordingly i.e. the 2nd args for Prewitt filter version 2
        imagemat = Prewitt_v2(filename)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

def convert_Prewitt_v1_all(BIRAD_0_dir, 
                           BIRAD_1_dir,
                           BIRAD_2_dir,
                           BIRAD_3_dir,
                           BIRAD_4A_dir,
                           BIRAD_4B_dir,
                           BIRAD_4C_dir,
                           BIRAD_5_dir,
                           destdir):
    converter_Prewitt_v1(BIRAD_0_dir,  destdir + '/BIRAD_0')
    converter_Prewitt_v1(BIRAD_1_dir,  destdir + '/BIRAD_1')
    converter_Prewitt_v1(BIRAD_2_dir,  destdir + '/BIRAD_2')
    converter_Prewitt_v1(BIRAD_3_dir,  destdir + '/BIRAD_3')
    converter_Prewitt_v1(BIRAD_4A_dir, destdir + '/BIRAD_4A')
    converter_Prewitt_v1(BIRAD_4B_dir, destdir + '/BIRAD_4B')
    converter_Prewitt_v1(BIRAD_4C_dir, destdir + '/BIRAD_4C')
    converter_Prewitt_v1(BIRAD_5_dir,  destdir + '/BIRAD_5')
    print("\n---edge detection completed--\n")


def convert_Prewitt_v2_all(BIRAD_0_dir, 
                           BIRAD_1_dir,
                           BIRAD_2_dir,
                           BIRAD_3_dir,
                           BIRAD_4A_dir,
                           BIRAD_4B_dir,
                           BIRAD_4C_dir,
                           BIRAD_5_dir,
                           destdir):
    converter_Prewitt_v2(BIRAD_0_dir,  destdir + '/BIRAD_0')
    converter_Prewitt_v2(BIRAD_1_dir,  destdir + '/BIRAD_1')
    converter_Prewitt_v2(BIRAD_2_dir,  destdir + '/BIRAD_2')
    converter_Prewitt_v2(BIRAD_3_dir,  destdir + '/BIRAD_3')
    converter_Prewitt_v2(BIRAD_4A_dir, destdir + '/BIRAD_4A')
    converter_Prewitt_v2(BIRAD_4B_dir, destdir + '/BIRAD_4B')
    converter_Prewitt_v2(BIRAD_4C_dir, destdir + '/BIRAD_4C')
    converter_Prewitt_v2(BIRAD_5_dir,  destdir + '/BIRAD_5')
    print("\n---edge detection completed--\n")


sourcedir = '/home/linh/Downloads/data/ori/BIRAD_4C'
destdir   = '/home/linh/Downloads/data/Prewitt_v1_preprocessed_data/BIRAD_4C'
os.makedirs(destdir, exist_ok=False)
print("The new directory is created!")
with Pool(28) as p:
    p.map(converter_Prewitt_v1(sourcedir, destdir))
#convert_edge_dir(sourcedir, destdir)

#sourcedir = '/home/linh/Downloads/data/ori/BIRAD_4C'
#destdir   = '/home/linh/Downloads/data/Prewitt_v2_preprocessed_data/BIRAD_4C'
#os.makedirs(destdir, exist_ok=False)
#print("The new directory is created!")
#with Pool(28) as p:
#    p.map(converter_Prewitt_v2(sourcedir, destdir))
#convert_edge_dir(sourcedir, destdir)