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
resize_width = 90
resize_height = 120

# image grid 크기
image_grid_width = 10 * resize_width
image_grid_height = 10 * resize_height

# image_grid 생성
image_grid = np.zeros((image_grid_height, image_grid_width, 3), dtype=np.uint8)

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
main_image = cv.resize(main_image, (900, 1200))

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
edge_image = cv.resize(main_image, (900, 1200))
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

mosaic_cartoon = (alpha * cartoon_image + (1 - alpha) * image_grid).astype(np.uint8)

cv.imwrite('edge.jpg', edge_image)
cv.imwrite('mosaic.jpg', mosaic_art)
cv.imwrite('cartoon.jpg', cartoon_image)
cv.imwrite('mosaiccartoon.jpg', mosaic_cartoon)