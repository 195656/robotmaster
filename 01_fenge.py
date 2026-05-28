import cv2
import numpy as np


img = cv2.imread('bed_pic.png')

if img is None:
    print("错误：无法读取图片，请检查路径！")
    exit()

img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)

#  将 BGR 转换为 HSV 颜色空间
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. 设定粉红色/红色的 HSV 阈值范围
# 注：图片中的床面偏向粉红/玫瑰红。在 HSV 中，红色和粉红色的 H（色调）通常在 140-180 之间
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([180, 255, 255])

# 4. 创建掩膜（Mask）
mask = cv2.inRange(hsv, lower_pink, upper_pink)

# 5. 形态学处理（去噪）：使用闭运算连接床面内部的小缝隙，开运算去除周围的微小杂色
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# 6. 位运算：将清理后的掩膜与原图进行“与”操作，提取出彩色床面
result = cv2.bitwise_and(img, img, mask=mask_cleaned)


# 原图
cv2.imshow('Step 1: Original Image', img)
cv2.waitKey(0)

# 步骤二：HSV 空间
cv2.imshow('Step 2: HSV Space', hsv)
cv2.waitKey(0)

# 步骤三：初步提取的二值化掩膜（包含了一些粗糙的边缘和噪点）
cv2.imshow('Step 3: Initial Mask', mask)
cv2.waitKey(0)

# 步骤四：形态学滤波后的干净掩膜
cv2.imshow('Step 4: Cleaned Mask', mask_cleaned)
cv2.waitKey(0)

cv2.imshow('Step 5: Final Result', result)
cv2.waitKey(0)

cv2.destroyAllWindows()