import cv2
import time
import numpy as np
import os
import Hand_traking_module as hdm

###################################
brushThickness = 32
eThickness = 70
###################################
folder_path = "header"
list_img = os.listdir(folder_path)
# print(list_img)
overList = []
for imgPath in list_img:
    image = cv2.imread(f'{folder_path}/{imgPath}')
    overList.append(image)
# print(len(overList))
'''
whenever we get our img from overlayList we will add it to the header
on the screen
'''
header = overList[0]
drawColor = (0,0,0)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = hdm.HandModule(detectionCon=0.5)
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0
while True:
    # import image and reverse it
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # find land mark or positions
    img = detector.findHand(img)
    lmlist = detector.findPosition(img, draw=False)

    # check which finger
    if len(lmlist) != 0:
        # tip of index and middle finger
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingerUP()

        # selection mode: two finger --> select
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            if y1 < 245:
                if 1 < x1 < 300:
                    header = overList[0]
                    drawColor = (0, 0, 0)
                elif 240 < x1 < 360:
                    header = overList[2]
                    drawColor = (255, 0, 0)
                elif 360 < x1 < 490:
                    header = overList[2]
                    drawColor = (0,0,255)
                elif 625 < x1 < 850:
                    header = overList[2]
                    drawColor = (34, 255, 0)
                elif 900 < x1 < 1280:
                    header = overList[1]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # drawing mode : index finger --> draw
        if fingers[1] and fingers[2] == False:
            xp, yp = 0, 0
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp, yp = x1, y1

    imgGrey = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGrey, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # setting header image
    img[0:245, 0:1280] = header
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("AI Paint @BadshaLaskar", img)
    cv2.imshow("Canvas", imgCanvas+imgInv)
    cv2.waitKey(1)
