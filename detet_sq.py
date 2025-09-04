import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    img1 = frame.copy()
    img2 = frame.copy()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)  # 增加模糊核大小

    # 二值化处理
    _, frame_thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY_INV)

    # 形态学操作（闭操作）以连接断开的边缘
    kernel = np.ones((7, 7), np.uint8)  # 增加核大小
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)
    # 筛选轮廓
    filtered_contours = []
    for contour in contours:
        # 过滤掉太小的轮廓
        if cv2.contourArea(contour) <600:
            continue

        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)

        # 多边形近似
        epsilon = 0.03 * perimeter  # 增加epsilon值，减少对噪声的敏感度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)

        # 检查是否为四边形
        if len(approx) == 4 and 1000 < area:
            # 转换为点数组
            points = approx.reshape(4, 2)

            # 使用更稳定的点排序方法
            # 计算边界矩形
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int8(box)

            # 计算中心点
            center = np.mean(box, axis=0)

            # 对点进行排序（基于极角）
            def sort_points_clockwise(points):
                # 计算中心点
                center = np.mean(points, axis=0)
                # 计算每个点相对于中心点的角度
                angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
                # 按角度排序
                sorted_indices = np.argsort(angles)
                # 返回排序后的点
                return points[sorted_indices]

            sorted_points = sort_points_clockwise(points)

            # 计算四边形的角度
            angles = []
            for i in range(4):
                # 获取三个连续的点
                pt1 = sorted_points[i]
                pt2 = sorted_points[(i + 1) % 4]
                pt3 = sorted_points[(i + 2) % 4]

                # 计算向量
                vec1 = pt1 - pt2
                vec2 = pt3 - pt2

                # 计算角度（弧度）
                angle_rad = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
                angle_deg = np.abs(np.degrees(angle_rad)) % 180

                # 确保角度在0-180度范围内
                if angle_deg > 90:
                    angle_deg = 180 - angle_deg

                angles.append(angle_deg)

            # 检查角度是否接近90度（允许±20度的误差）
            angle_threshold = 15  # 增加角度阈值
            valid_angles = all(70 <= angle <= 110 for angle in angles)  # 放宽角度范围

            # 如果角度有效，则处理这个四边形
            if valid_angles:
                filtered_contours.append(np.array(sorted_points, dtype=np.float32))
                cv2.drawContours(img2, [sorted_points], -1, (0, 255, 0), 2)
    cv2.imshow("Filtered Contours", frame_thresh)
    cv2.imshow("Original", img1)
    cv2.imshow("Contours", img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
