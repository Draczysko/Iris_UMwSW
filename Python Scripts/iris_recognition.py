from sys import maxsize
import numpy as np
from cv2 import cv2
from numpy.core.arrayprint import array2string
from numpy.core.defchararray import array, index, split
from numpy.lib.function_base import append
from daugman import daugman
from scipy.spatial import distance
import itertools
import glob
np.set_printoptions(threshold=maxsize)


def daugman_normalizaiton(image, height, width, r_in, r_out):
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    #r_out = r_in + r_out
    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
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


def load_baza(files):

    files = []
    for r in range(30):
        ar = np.array(r+1)
        cur_path = "UBIRIS_V1_800_600/Sessao_1/" + array2string(ar)
        i = 0
        for file in glob.glob(cur_path + "/*.jpg"):
            files.append(file)
            i+=1
            if(i > 2):
                break

    print(files.__len__())
    return files

def Search(kod):
    Codes = [code for code in glob.glob("Baza/*.jpg")]
    Hamm = []
    for code in Codes:

        Bits =  cv2.cvtColor(cv2.imread(code), cv2.COLOR_BGR2GRAY)
        th, Bits =  cv2.threshold(Bits, 0, 255, cv2.THRESH_BINARY)
        Hamm.append(distance.hamming(Bits.ravel(), kod.ravel()))
        

        #temp = np.zeros((h, w, z), np.uint8)
        # temp = FiltCopy[:,0,:]
        # FiltCopy[:,0,:] = FiltCopy[:,w-1,:]
        # FiltCopy[:,w-1,:] = temp
    Min = min(Hamm)
    
    MinM = Codes[Hamm.index(Min)]
    cv2.imshow("_Org", kod)
    cv2.imshow("_Found", cv2.imread(MinM))
    print(MinM)
    print(Min)

files =[]

#dla bazy
Baza = load_baza(files)

#dla porównania zdjęć z bazą
Files = [file for file in glob.glob("foty/*.jpg")]
names = []
dirs = []

files = Files


#files.append(cur_path + "/Img_2_1_4.jpg")
#files.append(cur_path + "/Img_1_1_2.jpg")
# files.append(cur_path + "/Img_2_1_2.jpg")
# files.append(cur_path + "/Img_2_1_3.jpg")
# file = cur_path + "/064R_2.png"

#images = load_images(files)
#bows = []


#----------------------------------------
#-----------GŁÓWNA PĘTLA-----------------
#----------------------------------------

NotFound = []

for file in files:
    print()
    img = cv2.imread(file, 0)
    img = cv2.medianBlur(img, 5)


    #Użycie adatptive threshold dla pozbycia się zbędnych wartości piskeli

    th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY, 35, 5)
    cv2.imshow(' ', th1)
    #cv2.waitKey(0)


    # Zamkniecie i otwracie usuwa większąść szumu, lepiej widać zarys oka

    morph = cv2.morphologyEx(th1, cv2.MORPH_OPEN, cv2.getStructuringElement(2, (5, 5)))
    cv2.imshow(' ', morph)
   # cv2.waitKey(0)

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, cv2.getStructuringElement(2, (3, 3)))
    cv2.imshow(' ', morph)
   # cv2.waitKey(0)

    # Po morfologii ładnie widać zarys źrenicy, wykorzystuje to do znalezienia jej przez Hough'a. 
    # Po odpowiednim dostosowaniu paramterów nawet ładnie działa.

    cimg = img.copy()

    # Wbudowana funkcja Hougha do znajdowania okręgów

    circles = cv2.HoughCircles(morph, cv2.HOUGH_GRADIENT_ALT, 2.5, 50, param1=100, param2=0.6, minRadius=25, maxRadius=60)
   
    height, width = img.shape
    r = 0
    mask = np.zeros((height, width), np.uint8)

    if circles is not None:
        for i in circles[0, :]:
            # print(i[2])
            cv2.circle(cimg, (i[0].astype(int), i[1].astype(int)), i[2].astype(int), (0, 0, 0), 3)
            cv2.circle(mask, (i[0].astype(int), i[1].astype(int)), i[2].astype(int), (255, 255, 255), thickness=3)
            blank_image = cimg[:int(i[1]), :int(i[1])]

            # Jeżeli znalazł źrenicę wykorzystuje Daugmana żeby znaleść zarys tęczówki. 
            # Działa dobrze jeżeli znamy prawdopodobny środek i promień. (Jest mocno obliczenio żerny)

            Cx = i[0].astype(int)
            Cy = i[1].astype(int)
            start_r = 90

            a = range(Cx - 3, Cx + 3, 2)
            b = range(Cy - 3, Cy + 3, 2)
            all_points = itertools.product(a, b)

            values = []
            coords = []

            
            for p in all_points:
                tmp = daugman(p, start_r, img)
                if tmp is not None:
                    val, circle = tmp
                    values.append(val)
                    coords.append(circle)

            #return the radius with biggest intensiveness delta on image
            #((xc, yc), radius)
            #x10 faster than coords[np.argmax(values)]
            center, radius = coords[values.index(max(values))]

            cv2.circle(cimg, center, radius, (0, 0, 0), 3)

            # wycinanie oka
            y0 = i[1].astype(int) - i[2].astype(int)
            y1 = i[1].astype(int) + i[2].astype(int)
            x0 = i[0].astype(int) - i[2].astype(int)
            x1 = i[0].astype(int) + i[2].astype(int)
            eye_img = cimg[y0:y1, x0:x1]
            

            # print(eye_img.shape)

            # masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)
            # _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # x, y, w, h = cv2.boundingRect(contours[0][0])
            # crop = masked_data[y:y+h, x:x+w]
            r = i[2].astype(int)
            #r2 = j[2]
        # cv2.imshow("edge", cimg)
        # cv2.waitKey(0)
        # cv2.imshow("edge", eye_img)
        # cv2.waitKey(0)
        # print(cimg.shape)

        #tworzenie wstęgi ze znalezionej tęczówki
        image_nor = daugman_normalizaiton(img, 90, 360, r, radius)
        # print(image_nor.shape)
        # cv2.imshow("edge", image_nor)
        # cv2.waitKey(0)

        #retval = cv.getGaborKernel(ksize, sigma, theta, lambd, gamma[, psi[, ktype]]    )


        #kilkuktrona filtracja
        As = [0,30,60,90,120,150]

        for i, A in enumerate(As):

            g_kernel = cv2.getGaborKernel((27, 27), 6.0, np.pi*A/180, 8.0, 0.8, 0, ktype=cv2.CV_32F)
            filtered_img = cv2.filter2D(image_nor, cv2.CV_8UC3, g_kernel)
            filtered_img += filtered_img
        
        filtered_img = filtered_img / filtered_img.max() *255
        filtered_img = filtered_img.astype(np.uint8)
        # plt.imshow(image_nor)
        # plt.imshow(filtered_img)
        # cv2.imshow("edge", filtered_img)
        # cv2.waitKey(0)


        cv2.imshow(' ', cimg)
        #cv2.waitKey(0)
        cv2.imshow(' 1', image_nor)
        # cv2.waitKey(0)
        cv2.imshow(' 2', filtered_img)
        #cv2.waitKey(0)

        filtered_img = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        ret, filtered_img = cv2.threshold(filtered_img, 0, 255, cv2.THRESH_BINARY)
        
        #print(filtered_img.shape)
        h, w= filtered_img.shape
        cv2.imshow(' 3', filtered_img)

        if Baza.count(file)==True:
            
            cv2.imwrite("Baza/kod"+array2string(np.array(files.index(file)))+".jpg",filtered_img)
            print(file)
        else:
            Search(filtered_img)
    else:
        NotFound.append(file)
        print(file + " ----> błąd")
        continue
cv2.waitKey(0)
#wyszukiwanie najlepszego dopasowania (przesywanie bitów)




# FiltCopy = filtered_img.copy()
# Min =[]

# for i in range(w):
    
#     Min.append(distance.hamming(bows[0].ravel(), FiltCopy.ravel()))

#     #temp = np.zeros((h, w, z), np.uint8)
#     temp = FiltCopy[:,0,:]
#     FiltCopy[:,0,:] = FiltCopy[:,w-1,:]
#     FiltCopy[:,w-1,:] = temp

#print(min(Min))
cv2.destroyAllWindows()
