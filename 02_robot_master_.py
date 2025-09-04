import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

detection_history = []
history_length = 5
stable_threshold = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取帧")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)

    _, frame_thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3, 3), np.uint8)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    draw_frame = frame.copy()
    result_frame = frame.copy()

    cv2.drawContours(draw_frame, contours, -1, (255, 0, 0), 2)

    # 筛选
    filtered_contours = []
    for contour in contours:
        # 过滤掉太小的轮廓
        if cv2.contourArea(contour) < 50:
            continue

        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)

        # 多边形近似
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)

        # 计算凸包
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)

        # 检查是否为凹多边形
        is_concave = not cv2.isContourConvex(approx)

        # 计算面积比
        area_ratio = area / hull_area if hull_area > 0 else 0

        # 计算边界矩形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h if h > 0 else 0

        if (7 <= len(approx) <= 9 and  # 顶点数范围
                1200< area < 5000 and # 面积范围
                is_concave and # 必须是凹多边形
                # 0.4 <= area_ratio <= 0.8 and  # 面积比范围
                0.7 <= aspect_ratio <= 1.3):  # 宽高比接近1

            # 计算轮廓的Hu矩
            moments = cv2.moments(approx)
            hu_moments = cv2.HuMoments(moments)

            # 对Hu矩取对数以便比较
            for i in range(7):
                hu_moments[i] = -1 * np.sign(hu_moments[i]) * np.log10(np.abs(hu_moments[i]))

            # 进一步筛选基于形状特征
            # 四角星应该有较大的Hu矩变化
            hu_variation = np.std(hu_moments)

            if hu_variation > 0.5:  # Hu矩变化较大，表明形状复杂
                filtered_contours.append(approx)

                # 在结果帧上绘制四角星
                cv2.drawContours(result_frame, [approx], -1, (0, 255, 0), 2)

                # 显示四角星信息
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    cv2.putText(result_frame, "STAR", (cX - 15, cY - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(result_frame, f"{len(approx)}pts", (cX - 15, cY + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # 更新检测历史
    detection_history.append(len(filtered_contours) > 0)
    if len(detection_history) > history_length:
        detection_history.pop(0)

    # 计算稳定检测次数
    stable_detection = sum(detection_history) >= stable_threshold

    # 只在稳定检测时显示结果
    if stable_detection:
        final_result = result_frame
    else:
        final_result = frame.copy()
        cv2.putText(final_result, "Detecting...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示处理结果
    cv2.imshow("Threshold", frame_thresh)
    cv2.imshow("All Contours", draw_frame)
    cv2.imshow("Result", final_result)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()