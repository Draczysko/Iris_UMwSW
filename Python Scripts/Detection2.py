import cv2
import matplotlib.pyplot as plt
import glob
import imutils
from PIL import Image
import numpy as np
from daugman import find_iris

print("Start")
cur_path = "foty"
x = 1
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def circuls(img_3):
    off = 2
    circleIn = cv2.HoughCircles(img_3,cv2.HOUGH_GRADIENT,1,100,
                                param1=30,param2=22,minRadius=10,maxRadius=30  )   #wbudowana funckja Hougha
    circleOut = cv2.HoughCircles(img_3,cv2.HOUGH_GRADIENT,1,150,
                                param1=50,param2=22,minRadius=60,maxRadius=120  )
    circleIn = np.uint16(np.around(circleIn))
    for i in circleIn[0,:]:
        # draw the outer circle
        cv2.circle(resized,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(resized,(i[0],i[1]),2,(0,0,255),3)
    
        circleOut = np.uint16(np.around(circleOut))
        for j in circleOut[0,:]: 
            d = int(i[0])-int(j[0])
            f = int(i[1])-int(j[1])
            if abs(d)<5 and abs(f)<5:

                (xc,yc) = (i[0],i[1])

                # draw the outer circle
                cv2.circle(resized,(i[0],i[1]),j[2]+off,(0,255,0),2)
                # draw the center of the circle
                cv2.circle(resized,(i[0],i[1]),2,(0,0,255),3)


    cv2.imshow('detected circles',resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Funkcja wyznaczenie krawędzi - algorytm Canny
def edge(img_1):
    img_1 = cv2.Canny(img_1, 20, 30)
    cv2.imshow("edge", img_1)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img_1


# Funkcja - znajdowanie konturów
def find_contours(img, img_2):
    img_c = img.copy()
    image_c = img_2.copy()

    # znalezienie konturów i posegregowanie ich
    contours = cv2.findContours(img_c, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]

    # Pentla dla znalezenia konturu spełniającego wymagania tablicy
    for i in contours:

        # aprksmymacja ilości lini składajacych się na kontur
        approx = cv2.approxPolyDP(i, 0.022 * cv2.arcLength(i, True), True)

        # Jeżeli są 4 linie to prawdopodobnie prostokąt
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            area = cv2.contourArea(approx)

            # nie ma sensu wykrywać obiektów mniejszych bądź większych od podanych
            if w > 60 and h > 10 and area > 2200 and area < 80000:
                ar = w / float(h)

                # przesiew kształtu przez współczynnik kształtu 
                if ar <= 0.30 or ar >= 3.33:
                    i = i.astype("float")
                    i *= ratio
                    i = i.astype("int")

                    # wyrysowanie konturu na obrazie
                    cv2.drawContours(img_2, [i], -1, (0, 255, 0), 3)
                    cv2.imshow("contours", img_2)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # przejście do funkji wycinania kształtu z obrazu
                    cut(i, image_c)
    return img_2


# wycinanie tablicy
def cut(x, img2):
    off = 2
    (a, b, c, d) = cv2.boundingRect(x)
    # warunek wyjścia za obrzeża obrazu przy wycinaniu
    if (b - off) < 0 or (b + d + off) > img2.shape[0] or (a - off) < 0 or (a + c + off) > img2.shape[1]:
        print("Wyjście za obraz!")
    else:
        # Wycięcie tablicy z obrazu
        tablica_cut = img2[(b - off):(b + d + off), (a - off):(a + c + off)]
        # cv2.imwrite("tablica.jpg",tablica_cut)
        cv2.imshow("tablica", tablica_cut)

        # rgb_cut = cv2.cvtColor(tablica_cut, cv2.COLOR_BGR2RGB)
        # cv2.imshow("szare",rgb_cut)
        # cv2.waitKey(0)

        # odczyt numerów
        text = pytesseract.image_to_string(tablica_cut, lang='eng',
                                           # znajdywanie jedynie konkretnych znaków
                                           config='--psm 8 --oem 3 -c tessedit_char_whitelist=-:0123456789ABCDEFGHIJKLM'
                                                  'NOPRSTUWYQVZX')

        print(text)


# Analiza zdjęc zebranych w folderze "foty":
#for file in glob.glob(cur_path + "/*.jpg"):
file = cur_path + "/062L_1.png"
print("Analizowanie zdjęcia - {}".format(file))
image = cv2.imread(file)

resized = imutils.resize(image.copy(), width=300)  # zmiana rozdzielczości aby ujednolicić obraz
ratio = image.shape[0] / float(resized.shape[0])
cv2.imshow("Zdjecie", resized)
cv2.waitKey(0)

#resized2 = resized.copy()[0:130, 0:130]

# Zdjęcie w skali szarości
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow("szare",gray)
cv2.waitKey(0)

# minimal_iris_radius = 10
# answer = find_iris(gray, minimal_iris_radius)
# print(answer)

# iris_center, iris_rad = answer

# # plot result
# out = resized2.copy()
# cv2.circle(out, iris_center, iris_rad, (0, 0, 255), 1)
# cv2.imshow("koniec",out)


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
circuls(blurred)
# Wyznaczenie obszaru tablicy rejestracyjnej
# tablica = find_contours(th2.copy(), image)

print("Koniec")
cv2.waitKey(0)
