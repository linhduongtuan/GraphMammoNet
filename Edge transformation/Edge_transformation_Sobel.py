import cv2, os, glob
from multiprocessing.pool import Pool


def Sobel_converter(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        # Read the original image
        img      = cv2.imread(filename,flags=0)
        image    = cv2.resize(image,(512, 512)) 
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img,(3,3), SigmaX=0, SigmaY=0)
        # Sobel Edge Detection
        #sobelx   = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        #sobely   = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
        sobelxy  = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', sobelxy) #create the edge image and store it to consecutive filenames
        filecnt + =1
    print("\n\n--saved in " + destdir + "--\n")

sourcedir = ('/home/linh/Downloads/data/ori/BIRAD_5')
destdir = ('/home/linh/Downloads/data/Sobel_preprocessed_data/BIRAD_5')
os.makedirs(destdir, exist_ok=False)
print("The new directory is created!")
with Pool(28) as p:
    p.map(Sobel_converter(sourcedir, destdir))
#convert_edge_dir(sourcedir, destdir)