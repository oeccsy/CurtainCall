import cv2 as cv
import numpy as np
import mediapipe as mp

# prepare hand tracking model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 카메라 장치 연결 및 출력 설정
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 30)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 30)


# 화면 출력
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    max_num_hands=1) as hands:

    while cv.waitKey(33) < 0:
        ret, original_frame = capture.read()
        edited_frame = original_frame.copy()
    
        # 카메라 입력이 가능한 경우에만 출력
        if not ret:
            break
    
        # applying hand tracking model => openCV는 BGR, mediapipe는 RGB 체제를 사용하므로 Convert 필요
        edited_frame = cv.cvtColor(edited_frame, cv.COLOR_BGR2RGB)
        hand_tracking_results = hands.process(edited_frame)
        edited_frame = cv.cvtColor(edited_frame, cv.COLOR_RGB2BGR)

        # drawing landmarks on the frame
        if hand_tracking_results.multi_hand_landmarks:
            for hand_landmarks in hand_tracking_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(edited_frame, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

        # 좌우 반전하여 거울처럼 출력 함
        edited_frame = cv.flip(edited_frame, 1)
        cv.imshow("Camera", edited_frame)

# Release
capture.release()
cv.destroyAllWindows()