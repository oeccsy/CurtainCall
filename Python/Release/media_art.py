import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from sklearn.cluster import KMeans

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

# image_grid 생성
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

# alpha blending 하여 모자이크 아트 구현
alpha = 0.7
mosaic_art = (alpha * main_image + (1 - alpha) * image_grid).astype(np.uint8) # Alternative) cv.addWeighted()

# edge detection으로 edge image 생성
threshold1 = 500
threshold2 = 250
aperture_size = 5

edge_image = cv.imread(f"./Resources/MainImage/{main_image_file_list[0]}", cv.IMREAD_GRAYSCALE)
edge_image = cv.resize(main_image, (1000, 1000))
gray_copy = edge_image.copy() # 추후 활용을 위해 복사
edge_image = cv.Canny(edge_image, threshold1, threshold2, apertureSize=aperture_size)
edge_image = cv.cvtColor(edge_image, cv.COLOR_GRAY2BGR) # subtract를 위해 BGR로 변환

# blur image 생성
blur_image = cv.medianBlur(main_image, 5)

# cartoon style image
# 클러스터링
pixel_values = main_image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 10
_, labels, centers = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
res = centers[labels.flatten()]
res = res.reshape((main_image.shape))

gray = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)
edges = cv.Laplacian(gray, cv.CV_8U, ksize=5)
ret, mask = cv.threshold(edges, 100, 255, cv.THRESH_BINARY_INV)
color = cv.bilateralFilter(res, 9, 250, 250)
cartoon_image = cv.bitwise_and(color, color, mask=mask)


# 출력 확인
#cv.imshow("test", edge_image)

# merge 출력 확인 (확인 완료)
#merge = np.hstack((main_image, image_grid, blend))
#cv2.imshow('Image Blending: Image1 | Image2 | Blended', merge)   

# 기존 이미지들을 덮어둘 이미지 생성
image_cover = np.full((1000, 1000, 3), 255, dtype=np.uint8) # subtract를 진행하여 검은색으로 덮는다.

# hand tracking 준비
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 애니메이션 처리를 위해 landmark를 저장할 변수
prev_landmark_list = [(-1, -1) for _ in range(200)]
cur_landmark_index = 0
limit_landmark = 200

# 렌더모드를 바꿀 변수
EDGE = 0
MOSAIC_ART = 1
BLUR = 2
CARTOON = 3
rendermode = 0

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

    while cv.waitKey(1) < 0:
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
                cv.circle(image_cover, (target_pos_x, target_pos_y), radius=60, color=(0, 0, 0), thickness=-1)
                
                # 이전 랜드마크 위치에 cover 추가
                prev_landmark_pos = prev_landmark_list[cur_landmark_index]
                if(prev_landmark_pos != (-1, -1)):
                  cv.circle(image_cover, (prev_landmark_pos[0], prev_landmark_pos[1]), radius=60, color=(255, 255, 255), thickness=-1)
                prev_landmark_list[cur_landmark_index] = (target_pos_x, target_pos_y)
                cur_landmark_index = cur_landmark_index + 1 
                if(cur_landmark_index >= limit_landmark):
                  cur_landmark_index = 0
                  # 손으로 100번 그린 경우 렌더모드 교체
                  rendermode = (rendermode + 1) % 4
        
        # 좌우 반전하여 거울처럼 출력 함
        edited_frame = cv.flip(edited_frame, 1)
        edited_frame = cv.resize(edited_frame, (1000, 1000))
      
        
        if rendermode == EDGE :
          edge_with_cover = cv.subtract(edge_image, image_cover)
          merge = np.hstack((edge_with_cover, edited_frame))
        elif rendermode == MOSAIC_ART:
          masaic_art_with_cover = cv.subtract(mosaic_art, image_cover)
          merge = np.hstack((masaic_art_with_cover, edited_frame))
        elif rendermode == BLUR:
          blur_with_cover = cv.subtract(blur_image, image_cover)
          merge = np.hstack((blur_with_cover, edited_frame))
        elif rendermode == CARTOON:
          cartoon_with_cover = cv.subtract(cartoon_image, image_cover)
          merge = np.hstack((cartoon_with_cover, edited_frame))
        cv.imshow("CurtainCall", merge)

#Release
capture.release()
cv.waitKey(0)
cv.destroyAllWindows()