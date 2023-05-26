import os
import cv2 as cv
import numpy as np
import mediapipe as mp

# 이미지가 저장된 폴더 경로
main_image_file_path = "./Resources/MainImage"
sub_images_file_path = "./Resources/SubImages"

# 폴더 내 모든 파일 리스트 가져오기
main_image_file_list = os.listdir(main_image_file_path)
sub_images_file_list = os.listdir(sub_images_file_path)

# 파일 리스트 확인
for file in main_image_file_list:
  print(f"main image : {file}")
  
for file in sub_images_file_list:
  print(f"sub image : {file}")
  
# 이미지 파일 확장자
image_extensions = [".jpg", ".jpeg", ".png"]

# main image load
main_image = cv.imread(f"./Resources/MainImage/{main_image_file_list[0]}")

# sub images load
sub_images_count = 0
sub_images_list = []

for file_name in sub_images_file_list:
  image = cv.imread(f"./Resources/SubImages/{file_name}")
  sub_images_list.append(image)
  sub_images_count += 1
  
  # 이미지 개수가 100개에 도달하면 종료
  if sub_images_count == 100:
    break

print(f"로드된 이미지 개수: {sub_images_count}")

# 개별 이미지 리사이즈 크기
resize_width = 100
resize_height = 100

# image grid 크기
image_grid_width = 10 * resize_width
image_grid_height = 10 * resize_height

# 출력 이미지 생성
image_grid = np.zeros((image_grid_height, image_grid_height, 3), dtype=np.uint8)

# 이미지를 가로로 10개, 세로로 10개씩 배치하여 출력 이미지에 병합
for i in range(100):
    # sub image resize
    resized_image = cv.resize(sub_images_list[i], (resize_width, resize_height))

    # 출력 이미지에 이미지 배치
    row = i // 10
    col = i % 10
    y_start = row * resize_height
    y_end = y_start + resize_height
    x_start = col * resize_width
    x_end = x_start + resize_width
    image_grid[y_start:y_end, x_start:x_end] = resized_image

# main image resize
main_image = cv.resize(main_image, (1000, 1000))

# 체크용 이미지 출력
#cv2.imshow("Output Image", image_grid)   # 출력 확인 완료
#cv2.imshow("Main Image", main_image)     # 출력 확인 완료


# alpha blending 하여 모자이크 아트 구현
alpha = 0.7
mosaic_art = (alpha * main_image + (1 - alpha) * image_grid).astype(np.uint8) # Alternative) cv.addWeighted()

# merge 출력 확인 (확인 완료)
#merge = np.hstack((main_image, image_grid, blend))
#cv2.imshow('Image Blending: Image1 | Image2 | Blended', merge)   

# 기존 이미지들을 덮어둘 이미지, 알파값으로 정보로 사용할 frame 생성
image_cover = np.full((1000, 1000, 3), 255, dtype=np.uint8) # subtract를 진행하여 검은색으로 덮는다.
image_alpha = np.full((1000, 1000), 100, dtype=np.uint8)

# hand tracking 준비
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 카메라 장치 연결 및 출력 설정
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 50)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 50)

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
            
            for hand_landmarks in hand_tracking_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(edited_frame, hand_landmarks, connections=mp_hands.HAND_CONNECTIONS)
                
                # 검지로 해당 위치의 cover 제거
                target_pos_x = 999 - int(hand_landmarks.landmark[8].x * 1000)
                target_pos_y = int(hand_landmarks.landmark[8].y * 1000)
                cv.circle(image_alpha, (target_pos_x, target_pos_y), radius=60, color=0, thickness=-1)
        
        # 좌우 반전하여 거울처럼 출력 함
        edited_frame = cv.flip(edited_frame, 1)
        
        # 모자이크 아트 프레임 작업
        masaic_art_with_cover = cv.subtract(mosaic_art, image_cover)
        
        # 랜드마크에 대응하는 픽셀은 cover를 지운다.
        #for distance in range(20):
        
        # merge 확인
        edited_frame = cv.resize(edited_frame, (1000, 1000))
        merge = np.hstack((masaic_art_with_cover, edited_frame))
        cv.imshow("test", merge)


#Release
capture.release()
cv.waitKey(0)
cv.destroyAllWindows()