import cv2
import numpy as np
from sympy.abc import epsilon



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

while True:
    filtered_contours = []
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取帧")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化处理
    _, frame_thresh = cv2.threshold(frame_gray, 125, 255, cv2.THRESH_BINARY_INV)

    cv2.imshow("frame_thresh",frame_thresh)
    #展示二值化后的图像，

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnt = contours[0]
    print('contours', len(contours))

    draw_frame = frame.copy()#展示轮廓
    rec_frame = frame.copy()#显示矩形框
    app_frame = frame.copy()#显示多边形
    chosen_frame = frame.copy()#显示最终结果
    poly_frame = frame.copy()#显示规则多边形

    res = cv2.drawContours(draw_frame, contours, -1, (255, 0, 0), 2)
    cv2.imshow("frame_contours",res)
    #展示能找到的轮廓
    approx_contours = []  # 存储近似后的轮廓
    regular_polygons = []  # 存储规则多边形
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        approx_contours.append(approx)
        area = cv2.contourArea(approx)
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(rec_frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        # 四角星检测条件
        if 7 <= len(approx) <= 9 and 150 < area < 300:
            # 凹多边形
            if not cv2.isContourConvex(approx):
                # 凸包
                hull = cv2.convexHull(approx)
                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    area_ratio = area / hull_area
                    if 0.3 <= area_ratio <= 0.9:
                        print(f"检测到四角星候选！顶点:{len(approx)}, 面积比:{area_ratio:.2f}")
                        filtered_contours.append(approx)
                        regular_polygons.append(approx)
                        cv2.drawContours(poly_frame, [approx], -1, (0, 255, 255), 2)  # 黄色

                        # 显示四角星信息
                        M = cv2.moments(approx)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])

                            cv2.putText(poly_frame, "STAR", (cX - 15, cY - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            cv2.putText(poly_frame, f"{len(approx)}pts", (cX - 15, cY + 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


                cv2.drawContours(poly_frame, [approx], -1, (0, 255, 0), 3)
    cv2.drawContours(chosen_frame, filtered_contours, -1, (255, 0, 0), 2)
    # cv2.imshow("rec_frame",rec_frame)
    # cv2.imshow("poly_frame",poly_frame)
    # res2 = cv2.drawContours(app_frame, contours, -1, (255, 0, 0), 2)
    # cv2.imshow("app",res2)
    # res3 = cv2.drawContours(chosen_frame, filtered_contours, -1, (255, 0, 0), 2)
    cv2.imshow("chosen_frame",chosen_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()