import cv2 as cv
import numpy as np
import mediapipe as mp

# 카메라 장치 연결 및 출력 설정
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 320)

# 화면 출력
while cv.waitKey(1) < 0:
    ret, original_frame = capture.read()
    edited_frame = original_frame.copy()
    
    # 카메라 입력이 가능한 경우에만 출력
    if not ret:
        break
    
    # 좌우 반전하여 거울처럼 출력 함
    edited_frame = cv.flip(edited_frame, 1)
    cv.imshow("Camera", edited_frame)

# Release
capture.release()
cv.destroyAllWindows()