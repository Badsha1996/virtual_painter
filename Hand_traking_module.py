import cv2
import mediapipe as mp
import time

'''
This is the module to check bare minimum requirement for hand-tracking software
* CV2 --> Open cv
* mp  --> Mediapipe
* time--> for checking the frame rate
'''


class HandModule:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.maxHands,
                                         self.detectionCon, self.trackCon)
        self.mp_draw = mp.solutions.drawing_utils
        self.tipsIds = [4, 8, 12, 16, 20]

    # this function will draw the hands on the webcam
    def findHand(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(imgRGB)

        if self.result.multi_hand_landmarks:
            for hand_landmark in self.result.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand_landmark,
                                                self.mp_hands.HAND_CONNECTIONS)
        return img

    # This function will give us the position of our hands and id of each position as a list
    def findPosition(self, img, handNo=0, draw=True):
        self.poslist = []
        if self.result.multi_hand_landmarks:
            myhand = self.result.multi_hand_landmarks[handNo]
            for id, land_mark in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(land_mark.x * w), int(land_mark.y * h)
                self.poslist.append([id, cx, cy])
                # if draw:
                #     cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
        return self.poslist

    def fingerUP(self):
        fingers = []
        # Thumb
        if self.poslist[self.tipsIds[0]][1] < self.poslist[self.tipsIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # rest of the fingers
        for id in range(1, 5):
            if self.poslist[self.tipsIds[id]][2] > self.poslist[self.tipsIds[id] - 2][2]:
                fingers.append(0)
            else:
                fingers.append(1)
        return fingers


def main():
    pre_time = 0
    cur_time = 0
    # Code for running the webcam
    video_cap = cv2.VideoCapture(0)
    # For testing purpose
    d = HandModule()
    while True:
        success, img = video_cap.read()
        # test purpose
        img = d.findHand(img)
        poslist = d.findPosition(img)
        if len(poslist) != 0:
            print(poslist[4])

        # FPS Detection in real time
        cur_time = time.time()
        fps = 1 / (cur_time - pre_time)
        pre_time = cur_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 3)
        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
