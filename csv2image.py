from pandas import read_csv
from pandas import DataFrame
import numpy as np
import csv
import cv2
from PIL import Image as im

def convert2image(img):
    img = img.reshape(6,9)  # dimensions of the image
    image = np.zeros((6,9,3))  # empty matrix
    image[:,:,0] = img
    image[:,:,1] = img
    image[:,:,2] = img
    return image.astype(np.uint8) # return the image
    
    
def csvtoImage():
    data2 = read_csv('NEWDATASET_malware.csv')
    feat2 = data2.keys()
    feat_labels2 = np.asarray(feat2)
    csvdataset = data2.values
    csvdataset = csvdataset[0:15500,:]
    count = 0
    for row in csvdataset:
        #img = convert2image(csvdataset[count])
        #cv2.imwrite('IMAGES_6x9/malware' + str(count) + '.jpg',img)
        count += 1
        data = DataFrame.to_numpy(row)
        data_image = im.fromarray(data)
        fp = 'IMAGES_6x9/malware/' + str(count) + '.jpg'
        data_image.save(fp)
        


if __name__ == "__main__":
    csvtoImage()
    
