import cv2
import matplotlib.pyplot as plt
import glob
import imutils
from PIL import Image
import numpy as np
from daugman import find_iris
from LogGabor import LogGabor
parameterfile = 'https://raw.githubusercontent.com/bicv/LogGabor/master/default_param.py'

lg = LogGabor(parameterfile)
sf_0 = .02 # TODO .1 cycle / pixel (Geisler)
params= {'sf_0':sf_0, 'B_sf': lg.pe.B_sf, 'theta':0., 'B_theta': lg.pe.B_theta}


print("Start")
cur_path = "foty"
x = 1


def circuls(img_3):
    img_C = resized.copy()
    off = 2
    circleIn = cv2.HoughCircles(img_3,cv2.HOUGH_GRADIENT,1.2,100,
                                param1=30,param2=22,minRadius=15,maxRadius=50  )   #wbudowana funckja Hougha
    circleOut = cv2.HoughCircles(img_3,cv2.HOUGH_GRADIENT,1,150,
                                param1=50,param2=22,minRadius=60,maxRadius=120  )
    circleIn = np.uint16(np.around(circleIn))
    for i in circleIn[0,:]:
        # draw the outer circle
        img_C = cv2.circle(img_C,(i[0],i[1]),i[2],(0,255,0),2)
        r1 = i[2]
        # draw the center of the circle
        img_C = cv2.circle(img_C,(i[0],i[1]),2,(0,0,255),3)
    
        circleOut = np.uint16(np.around(circleOut))
        for j in circleOut[0,:]: 
            d = int(i[0])-int(j[0])
            f = int(i[1])-int(j[1])
            if abs(d)<5 and abs(f)<5:
                
                r2 = j[2]
                (xc,yc) = (j[0],j[1])
                
                # draw the outer circle
                img_C = cv2.circle(img_C,(j[0],j[1]),j[2]+off,(0,255,0),2)
                # draw the center of the circle
                img_C = cv2.circle(img_C,(j[0],j[1]),2,(0,0,255),3)

    # minimal_iris_radius = r1 + 20
    # answer = find_iris(gray, minimal_iris_radius, xc, yc)
    # print(answer)

    # iris_center, iris_rad = answer

    # # plot result
    # out = resized.copy()
    # cv2.circle(out, iris_center, iris_rad, (0, 0, 255), 1)
    # cv2.imshow("koniec",out)

    cv2.imshow('detected circles',img_C)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return r1, r2, xc, yc



# Funkcja wyznaczenie krawędzi - algorytm Canny
def edge(img_1):
    img_1 = cv2.Canny(img_1, 20, 30)
    cv2.imshow("edge", img_1)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img_1



# liang's code for reference
def daugman_normalizaiton(image, height, width, r_in, r_out, Cx, Cy):       
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
   # r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
    circle_x = Cx
    circle_y = Cy

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



# Analiza zdjęc zebranych w folderze "foty":
#for file in glob.glob(cur_path + "/*.jpg"):
file = cur_path + "/Img_2_1_1.jpg"
print("Analizowanie zdjęcia - {}".format(file))
image = cv2.imread(file)

resized = imutils.resize(image.copy(), width=400)  # zmiana rozdzielczości aby ujednolicić obraz
ratio = image.shape[0] / float(resized.shape[0])
cv2.imshow("Zdjecie", resized)
cv2.waitKey(0)

#resized2 = resized.copy()[0:130, 0:130]

# Zdjęcie w skali szarości
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("szare",gray)
cv2.waitKey(0)

# Operacja rozmycia obrazu
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("blurred", blurred)
cv2.waitKey(0)

cannyGray = edge(blurred.copy())

# Operacja progowania globalnego
# ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("progowanie",th1)
# cv2.waitKey(0)

# # Operacja progowania adptacyjnego
# th1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                             cv2.THRESH_BINARY_INV, 9, 2)
# cv2.imshow("progowanie_2",th1)
# cv2.waitKey(0)



# # Operacja morfologiczna otwarcia
# th2 = cv2.morphologyEx(th1.copy(), cv2.MORPH_OPEN, cv2.getStructuringElement(2, (3, 3)))
# cv2.imshow("Otwarcie", th2)
# cv2.waitKey(0)

# th3 = cv2.morphologyEx(th2.copy(), cv2.MORPH_DILATE, cv2.getStructuringElement(2, (4, 3)))
# cv2.imshow("zamkniecie", th3)
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# Przedstawienie konturów w obrazie
#image2 = edge(th2.copy())

Crls = circuls(blurred)
r_in, r_out, Cx, Cy = Crls

image_nor = daugman_normalizaiton(blurred, 85, 360, r_in, r_out, Cy, Cx)
cv2.imshow("wstega",image_nor)
cv2.waitKey(0)


g_kernel = cv2.getGaborKernel((25, 25), 4.7, np.pi/5, 9.5, 0.40, 0, ktype=cv2.CV_32F)
filtered_img = cv2.filter2D(image_nor, cv2.CV_8UC3, g_kernel)

# filtered_img =lg.FTfilter(image_nor,B_sf)
# lg.loggabor_image()

cv2.imshow("filtr", filtered_img)

# h, w = g_kernel.shape[:2]
# g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)

# Wyznaczenie obszaru tablicy rejestracyjnej
# tablica = find_contours(th2.copy(), image)

print("Koniec")
cv2.waitKey(0)
