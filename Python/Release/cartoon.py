import cv2
import numpy as np
from sklearn.cluster import KMeans

# 이미지 파일을 읽어온다.
img_file = './Resources/Test/carrot.png'
img = cv2.imread(img_file)

# 클러스터링
pixel_values = img.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 10
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

centers = np.uint8(centers)
res = centers[labels.flatten()]
res = res.reshape((img.shape))

# 그레이 스케일로 변환한다.
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# 블러처리 한다.
gray = cv2.medianBlur(gray, 5)

# 엣지 검출을 수행한다.
edges = cv2.Laplacian(gray, cv2.CV_8U, ksize=5)
ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

# 컬러화한다.
color = cv2.bilateralFilter(res, 9, 250, 250)

# 카툰 렌더링을 수행한다.
cartoon = cv2.bitwise_and(color, color, mask=mask)

# 결과를 출력한다.
cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()