import cv2
import mediapipe as mp
import time

'''
# For frame rate detection
pre_time = 0
cur_time = 0

# Code for running the webcam
video_cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
# @hands class only uses RGB images so
# we have to convert it later in @imgRGB var
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = video_cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    # Test for checking value
    # print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            for id, land_mark in enumerate(hand_landmark.landmark):
                print(id, land_mark)
                h, w, c = img.shape
                cx, cy = int(land_mark.x*w), int(land_mark.y*h)
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

    # FPS Detection in real time
    cur_time = time.time()
    fps = 1 / (cur_time - pre_time)
    pre_time = cur_time

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
    '''