import cv2
import mediapipe as mp
from controller import Controller

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    img = cv2.flip(img, 1)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        Controller.hand_Landmarks = results.multi_hand_landmarks[0]
        mpDraw.draw_landmarks(img, Controller.hand_Landmarks, mpHands.HAND_CONNECTIONS)

        Controller.update_fingers_status()
        Controller.cursor_moving()
        Controller.detect_scrolling()
        Controller.detect_zoomming()  # Correct method name
        Controller.detect_clicking()
        Controller.detect_dragging()

    cv2.imshow('Hand Tracker', img)
    if cv2.waitKey(5) & 0xff == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
