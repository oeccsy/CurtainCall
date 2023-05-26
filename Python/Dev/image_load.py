import os
import cv2
import numpy as np
import time

# 이미지가 저장된 폴더 경로
folder_path = "./Resources"

# 폴더 내 모든 파일 리스트 가져오기
file_list = os.listdir(folder_path)

for file in file_list:
  print(file)

# 이미지 파일 확장자
image_extensions = [".jpg", ".jpeg", ".png"]

# 이미지 개수 카운터
image_count = 0

# 이미지를 저장할 리스트
images = []

# 폴더 내 파일들을 순회하며 이미지를 로드
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    # 파일이 이미지 파일인지 확인
    if os.path.isfile(file_path) and any(file_name.lower().endswith(ext) for ext in image_extensions):
        # 이미지 로드
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)
            image_count += 1

        # 이미지 개수가 100개에 도달하면 종료
        if image_count == 100:
            break

print(f"로드된 이미지 개수: {image_count}")
