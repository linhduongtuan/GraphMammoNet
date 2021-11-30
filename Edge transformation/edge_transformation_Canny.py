import cv2
import os
import glob
import time
from multiprocessing.pool import Pool


def Canny_converter(sourcedir, destdir):
    print("\n\n---reading directory " + sourcedir + "---\n")
    filecnt = 1
    for filename in glob.glob(sourcedir + '/*'):
        image    = cv2.imread(filename)
        image    = cv2.resize(image, (512, 512))
        imagemat = cv2.Canny(image, 100, 200)
        cv2.imwrite(destdir+'/img-'+str(filecnt)+'.png', imagemat) #create the edge image and store it to consecutive filenames
        filecnt += 1
    print("\n\n--saved in " + destdir + "--\n")

start = time.time()

sourcedir = ('/home/linh/Downloads/data/ori/BIRAD_3')
destdir = ('/home/linh/Downloads/data/Canny_preprocessed_data/BIRAD_3')
os.makedirs(destdir, exist_ok=True)
print("The new directory is created!")
#with Pool(28) as p:
#    p.map(Canny_converter(sourcedir, destdir))
Canny_converter(sourcedir, destdir)

end = time.time()
time_to_transform = (end - start)/60
print("Total time (min) for transforming edege :", time_to_transform)
print("=======End transforming edege process here======")
