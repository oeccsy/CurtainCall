import cv2 as cv
import numpy as np
import mediapipe as mp
import socket
import struct

# prepare hand tracking model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 카메라 장치 연결 및 출력 설정
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 30)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 30)

# UDP settings
UDP_IP = '127.0.0.1'
UDP_PORT = 5005


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
            hand_landmarks = hand_tracking_results.multi_hand_landmarks[0]
            
            # send position via UDP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            # test code
            '''
            sock.sendto(struct.pack('fff',
                                    hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z),
                                    (UDP_IP, UDP_PORT))

            '''
            
            sock.sendto(struct.pack('fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff',
                                    hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z,
                                    hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z,
                                    hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y, hand_landmarks.landmark[2].z,
                                    hand_landmarks.landmark[3].x, hand_landmarks.landmark[3].y, hand_landmarks.landmark[3].z,
                                    hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z,
                                    hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z,
                                    hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z,
                                    hand_landmarks.landmark[7].x, hand_landmarks.landmark[7].y, hand_landmarks.landmark[7].z,
                                    hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z,
                                    hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z,
                                    hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z,
                                    hand_landmarks.landmark[11].x, hand_landmarks.landmark[11].y, hand_landmarks.landmark[11].z,
                                    hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z,
                                    hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z,
                                    hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y, hand_landmarks.landmark[14].z,
                                    hand_landmarks.landmark[15].x, hand_landmarks.landmark[15].y, hand_landmarks.landmark[15].z,
                                    hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y, hand_landmarks.landmark[16].z,
                                    hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z,
                                    hand_landmarks.landmark[18].x, hand_landmarks.landmark[18].y, hand_landmarks.landmark[18].z,
                                    hand_landmarks.landmark[19].x, hand_landmarks.landmark[19].y, hand_landmarks.landmark[19].z,
                                    hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z),
                                    (UDP_IP, UDP_PORT))
            
            for hand_landmarks in hand_tracking_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(edited_frame, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)

        # 좌우 반전하여 거울처럼 출력 함
        edited_frame = cv.flip(edited_frame, 1)
        cv.imshow("Camera", edited_frame)

# Release
capture.release()
cv.destroyAllWindows()