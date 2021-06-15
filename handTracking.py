import cv2
import mediapipe as mp


capture = cv2.VideoCapture(0)

##hand detection module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils



while True:
    success, img = capture.read()
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            ##getting landamrk information
            for id, landmark in enumerate(hand.landmark):
                #print(id, landmark)
                height, width, channel = img.shape
                cx, cy  = int(landmark.x*width), int(landmark.y*height)
                #print(id, cx, cy)
                cv2.circle(img, (cx,cy), 15, (255,0,0), cv2.FILLED)
                #using ifs and ID one should be able to map for only one point and manage
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)



    cv2.imshow('Image', img)
    cv2.waitKey(1)
