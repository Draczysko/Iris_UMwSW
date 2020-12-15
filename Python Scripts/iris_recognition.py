import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage import io
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import cv2
import pandas as pd
from scipy.spatial import distance


def daugman_normalizaiton(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height,width, 3), np.uint8)
    circle_x = int(image.shape[0] / 2)
    circle_y = int(image.shape[1] / 2)

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)

            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat


def load_images(files):
    images = []
    for file in files:
        img = cv2.imread(file, 0)
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        images.append([img, cimg])
    return images

cur_path = "foty"
files = []
files.append(cur_path + "/Img_2_1_4.jpg")
files.append(cur_path + "/Img_2_1_1.jpg")
# file = cur_path + "/064R_2.png"
images = load_images(files)
bows = []
for file in files:
    img = cv2.imread(file, 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT_ALT, 1, 10, param1=100, param2=0.6, minRadius=100, maxRadius=0)
    print(circles.shape)
    height, width = img.shape
    r = 0
    mask = np.zeros((height, width), np.uint8)
    for i in circles[0, :]:
        print(i[2])
        cv2.circle(cimg, (i[0].astype(int), i[1].astype(int)), i[2].astype(int), (0, 0, 0))
        cv2.circle(mask, (i[0].astype(int), i[1].astype(int)), i[2].astype(int), (255, 255, 255), thickness=0)
        blank_image = cimg[:int(i[1]), :int(i[1])]

        masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0][0])
        crop = masked_data[y:y+h, x:x+w]
        r = i[2]
    # cv2.imshow("edge", cimg)
    # cv2.waitKey(0)
    print(cimg.shape)
    image_nor = daugman_normalizaiton(cimg, 60, 360, r, 55)
    # cv2.imshow("edge", image_nor)
    # cv2.waitKey(0)

    #retval = cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]    )
    g_kernel = cv2.getGaborKernel((27, 27), 8.0, np.pi/4, 10.0, 0.8, 0, ktype=cv2.CV_32F)
    filtered_img = cv2.filter2D(image_nor, cv2.CV_8UC3, g_kernel)

    # plt.imshow(image_nor)
    # plt.imshow(filtered_img)
    cv2.imshow("edge", filtered_img)
    cv2.waitKey(0)

    bows.append(filtered_img)
    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)

print(distance.hamming(bows[0].ravel(), bows[1].ravel()))
