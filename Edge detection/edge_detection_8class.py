# -*- coding: utf-8 -*-
"""Edge_detection_4class.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-hezbJLLMCvvdTDhbLAUzylihWARwvTX
"""

#mounting on drive
#from google.colab import drive
#drive.mount('/content/drive',force_remount=True)

import numpy as np
import cv2
import matplotlib.pyplot as plt
#from google.colab.patches import cv2_imshow
import glob


#applying filter on a single image
def apply_filter(filename,filter):
  print("reading file---> "+str(filename))
  img=cv2.imread(filename,cv2.IMREAD_GRAYSCALE)

  #img=cv2.resize(img,(250,250))
  #comment out the above line if there is memory issue i.e. need to resize all images to smaller dim

  h, w = img.shape
  print("shape: "+str(h)+" x "+str(w)+"\n")

  # define filters
  horizontal = filter
  vertical = np.transpose(filter)

  # define images with 0s
  newhorizontalImage = np.zeros((h, w))
  newverticalImage = np.zeros((h, w))
  newgradientImage = np.zeros((h, w))

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

#function for creating all edge-images of a directory
def convert_edge_dir(sourcedir,destdir):
  print("\n\n---reading directory "+sourcedir+"---\n")
  filecnt=1
  for filename in glob.glob(sourcedir+'/*'):
    #applying Prewitt filter
    #for appyling any other filter change filter value accordingly i.e. the 2nd args for apply_filter()
    imagemat=apply_filter(filename,np.array([[-1,0,1],[-1,0,1],[-1,0,1]]))
    cv2.imwrite(destdir+'/img-'+str(filecnt)+'.jpg',imagemat) #create the edge image and store it to consecutive filenames
    filecnt+=1
  print("\n\n--saved in "+destdir+"--\n")


#function for creating all edge-images under covid, non-covid, penumonia directories 
#since 4-class so COVID, NORMAL, Bacterial and Viral PNEUMONIA present
def convert_edge_all_dir(BIRAD_0_dir, 
                         BIRAD_1_dir,
                         BIRAD_2_dir,
                         BIRAD_3_dir,
                         BIRAD_4A_dir,
                         BIRAD_4B_dir,
                         BIRAD_4C_dir,
                         BIRAD_5_dir,
                         destdir):
  convert_edge_dir(BIRAD_0_dir, destdir + '/BIRAD_0')
  convert_edge_dir(BIRAD_1_dir, destdir + '/BIRAD_1')
  convert_edge_dir(BIRAD_2_dir, destdir + '/BIRAD_2')
  convert_edge_dir(BIRAD_3_dir, destdir + '/BIRAD_3')
  convert_edge_dir(BIRAD_4A_dir, destdir + '/BIRAD_4A')
  convert_edge_dir(BIRAD_4B_dir, destdir + '/BIRAD_4B')
  convert_edge_dir(BIRAD_4C_dir, destdir + '/BIRAD_4C')
  convert_edge_dir(BIRAD_5_dir, destdir + '/BIRAD_5')
  print("\n---edge detection completed--\n")

#adjust the source paths accordingly
datasetdir='Mammograms/data'
BIRAD_0_dir=datasetdir+'/BIRAD_0'
BIRAD_1_dir=datasetdir+'/BIRAD_1'
BIRAD_2_dir=datasetdir+'/BIRAD_2'
BIRAD_3_dir=datasetdir+'/BIRAD_3'
BIRAD_4A_dir=datasetdir+'/BIRAD_4A'
BIRAD_4B_dir=datasetdir+'/BIRAD_4B'
BIRAD_4C_dir=datasetdir+'/BIRAD_4C'
BIRAD_5_dir=datasetdir+'/BIRAD_5'


#adjust the destination directory accordingly
#directory naming format--> <dataset_name(not necessary conventional)>_edge/<filtername>
destdir='Mammograms/data'

convert_edge_all_dir(BIRAD_0_dir, 
                    BIRAD_1_dir,
                    BIRAD_2_dir,
                    BIRAD_3_dir,
                    BIRAD_4A_dir,
                    BIRAD_4B_dir,
                    BIRAD_4C_dir,
                    BIRAD_5_dir,
                    destdir)
