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

# 체크용 이미지 출력
#cv2.imshow("Output Image", image_grid)   # 출력 확인 완료
#cv2.imshow("Main Image", main_image)     # 출력 확인 완료


# alpha blending 하여 모자이크 아트 구현
alpha = 0.7
mosaic_art = (alpha * main_image + (1 - alpha) * image_grid).astype(np.uint8) # Alternative) cv.addWeighted()

# edge detection으로 edge image 생성
threshold1 = 500
threshold2 = 250
aperture_size = 5

edge_image = cv.imread(f"./Resources/MainImage/{main_image_file_list[0]}", cv.IMREAD_GRAYSCALE)
edge_image = cv.resize(main_image, (1000, 1000))
edge_image = cv.Canny(edge_image, threshold1, threshold2, apertureSize=aperture_size)

cv.imshow("test", edge_image)



cv.waitKey(0)
cv.destroyAllWindows()