import cv2
import numpy as np
from sympy.abc import epsilon


def cv_show(name,img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

template_path = r"E:\python\open_cv\robotmaster\master.jpg"

img = cv2.imread(template_path)
img=cv2.resize(img,(640,480))

approx_contours = [] #储存近似轮廓
drawed_img = img.copy()#画轮廓
rec_frame = img.copy()#框选后的轮廓
chosen_frame = img.copy()
filtered_contours = [] #储存筛选轮廓

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化处理
_, img_thresh = cv2.threshold(img_gray, 125, 255, cv2.THRESH_BINARY)
cv_show("img_thresh",img_thresh)
#展示二值化后的图像
# kernel_dilate = np.ones((5,3), np.uint8)  # 合并右上灯条的不连续部分
# dilated = cv2.dilate(img_thresh, kernel_dilate, iterations=1)
# dilated = cv2.morphologyEx(dilated,cv2.MORPH_CLOSE,kernel_dilate)
# cv_show("closeing",dilated)
# 找轮廓
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print("contours:",len(contours))
res1 = cv2.drawContours(drawed_img, contours, -1, (0, 0, 255), 2)
cv_show("drawed_img",res1)
for contour in contours:
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)
    area = cv2.contourArea(approx)
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(rec_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    if (5 < len(approx) < 7 and 800 < area < 1500) or (5 < len(approx) < 7 and 520< area < 550) :
        filtered_contours.append(approx)
    cv2.drawContours(rec_frame, [approx], -1, (0, 255, 0), 2)#画近似轮廓
cv2.drawContours(chosen_frame, filtered_contours, -1, (0, 255, 0), 2)# 画近似轮廓
cv_show("rec_frame",rec_frame)
cv_show("filtered_contours",chosen_frame)
