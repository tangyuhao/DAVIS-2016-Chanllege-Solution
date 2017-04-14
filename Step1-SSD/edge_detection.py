import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_edges(img_path):
    '''
    input: the image path
    output: none
    function: show the input image and the edges
    ''' 
    img = cv2.imread(img_path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((5,5),np.float32)/25
    # dst = cv2.filter2D(img,-1,kernel)


    edges = cv2.Canny(gray_image,100,200)

    plt.subplot(121),plt.imshow(RGB_img,cmap='gray',vmin=0,vmax=255)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def get_edges(img_path):
    '''
    input: the image path
    output: a numpy ndarray of the edges in this image 
    '''

    img = cv2.imread(img_path)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((5,5),np.float32)/25
    # dst = cv2.filter2D(img,-1,kernel)


    edges = cv2.Canny(gray_image,100,200)
    return edges


if __name__ == "__main__":
    show_edges("demo/000001.jpg")
    show_edges("demo/000008.jpg")
    show_edges("demo/000010.jpg")
    show_edges("demo/000022.jpg")