import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

def create_bounding_box(img_path, outfile_name, random_noise = True):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary_img = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
    

    # show the binary_image 
    # plt.subplot(1,1,1), plt.imshow(binary_img,'gray')
    # plt.show()
    image_cnt, contours, _ = cv2.findContours(binary_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])
    if (random_noise == True):
        w_org = w
        h_org = h
        x = int(np.random.normal(x, w_org * 0.15, 1))
        y = int(np.random.normal(y, h_org * 0.15, 1))
        w = int(np.random.normal(w, w_org * 0.15, 1))
        h = int(np.random.normal(h, h_org * 0.15, 1))
    height, width, channels = img.shape
    out_img = np.zeros((height,width,1), np.uint8)
    cv2.rectangle(out_img,(x,y),(x+w,y+h),(255,0,0),-1)
    cv2.imwrite(outfile_name, out_img)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_path = sys.argv[1]
        outfile_name = sys.argv[2]
        create_bounding_box(img_path, outfile_name, True)
