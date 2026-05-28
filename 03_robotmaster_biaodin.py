import cv2
import numpy as np


def sort_points_counter_clockwise(points):
    """改进的点排序函数，确保正确的3D-2D对应"""
    # 按x+y排序找到左上角点
    sum_coords = points.sum(axis=1)
    top_left_idx = np.argmin(sum_coords)

    # 按x-y排序找到右上角点
    diff_coords = np.diff(points, axis=1).flatten()
    top_right_idx = np.argmax(diff_coords)

    # 重新排序点 [左上, 右上, 右下, 左下]
    sorted_points = np.roll(points, -top_left_idx, axis=0)

    # 确保正确的顺序
    if np.linalg.norm(sorted_points[1] - sorted_points[2]) > np.linalg.norm(sorted_points[0] - sorted_points[1]):
        # 如果是顺时针，反转顺序
        sorted_points = sorted_points[::-1]

    return sorted_points

# 相机内参矩阵
camera_matrix = np.array([[812.74353412, 0., 999.84368755],
                          [0., 811.58454708, 504.75250109],
                          [0., 0., 1.]])

# 畸变系数
dist_coeffs = np.array([-1.99194847e-02, -2.54440885e-04, -1.41682184e-03,
                        2.51937464e-03, 1.43636328e-05])

# 定义矩形的3D世界坐标点（单位：厘米），左下角为原点
rect_length = 16  # 长
rect_width = 10  # 宽
object_points = np.array([
    [-rect_length / 2, -rect_width / 2, 0],  # 左下
    [rect_length / 2, -rect_width / 2, 0],  # 右下
    [rect_length / 2, rect_width / 2, 0],  # 右上
    [-rect_length / 2, rect_width / 2, 0]  # 左上
], dtype=np.float32)

# 坐标轴长度
axis_length = max(rect_length, rect_width) * 0.5
axis_points = np.array([
    [0, 0, 0],  # 原点
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, axis_length]  # Z轴 (垂直于纸面向外)
], dtype=np.float32)

cap = cv2.VideoCapture(0)
prev_rvec = None
prev_tvec = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img1 = frame.copy()
    img2 = frame.copy()
    img3 = frame.copy()  # 用于显示PnP结果的图像

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 去噪
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    # 二值化处理
    _, frame_thresh = cv2.threshold(blurred, 125, 255, cv2.THRESH_BINARY_INV)

    # 画线容易有空隙，闭操作
    kernel = np.ones((7, 7), np.uint8)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    h, w = frame_thresh.shape
    cv2.rectangle(frame_thresh, (0, 0), (w - 1, h - 1), 0, 1)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

    filtered_contours = []
    for contour in contours:
        zhouchang = cv2.arcLength(contour, True)

        # 多边形近似
        epsilon = 0.03 * zhouchang
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)

        # 检查是否为四边形
        if len(approx) == 4 and 200 < area and cv2.isContourConvex(approx):
            points = approx.reshape(4, 2)

            # 检查长宽比
            rect = cv2.minAreaRect(approx)
            width, height = rect[1]
            aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0

            # 检查角度
            valid_angles = True
            for i in range(4):
                pt1 = points[i]
                pt2 = points[(i + 1) % 4]
                pt3 = points[(i + 2) % 4]

                vec1 = pt1 - pt2
                vec2 = pt3 - pt2

                angle = np.abs(np.degrees(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])))
                angle = min(angle, 360 - angle)

                if not (70 <= angle <= 110):
                    valid_angles = False
                    break

            # 综合所有条件
            if valid_angles and 0.5 <= aspect_ratio <= 2.0:
                sorted_points = sort_points_counter_clockwise(points)
                filtered_contours.append(sorted_points)
                cv2.drawContours(img2, [sorted_points.astype(int)], -1, (0, 255, 0), 2)

    # 对每个检测到的矩形进行PnP计算
    for image_points in filtered_contours:
        # 确保图像点格式正确
        image_points_reshaped = image_points.reshape(-1, 1, 2).astype(np.float32)

        # 使用前一帧的rvec和tvec作为初始估计
        if prev_rvec is not None and prev_tvec is not None:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points_reshaped,
                camera_matrix,
                dist_coeffs,
                rvec=prev_rvec,
                tvec=prev_tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        else:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points_reshaped,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if success:
            # 保存当前解供下一帧使用
            prev_rvec = rvec.copy()
            prev_tvec = tvec.copy()

            # 计算重投影误差
            reprojected_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            reprojected_points = reprojected_points.reshape(-1, 2)

            # 计算重投影误差（像素距离）
            error = np.mean(np.linalg.norm(image_points - reprojected_points, axis=1))

            # 如果重投影误差太大，跳过这个检测结果
            if error > 30.0:  # 2像素阈值
                print(f"跳过检测结果，重投影误差太大: {error:.2f}")
                continue

            # 投影坐标轴到图像平面
            projected_axis_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_axis_points = projected_axis_points.reshape(-1, 2).astype(int)

            # 绘制坐标轴
            origin = tuple(projected_axis_points[0])
            # X轴 (红色)
            cv2.line(img3, origin, tuple(projected_axis_points[1]), (0, 0, 255), 3)
            # Y轴 (绿色)
            cv2.line(img3, origin, tuple(projected_axis_points[2]), (0, 255, 0), 3)
            # Z轴 (蓝色)
            cv2.line(img3, origin, tuple(projected_axis_points[3]), (255, 0, 0), 3)

            # 添加坐标轴标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img3, "X", tuple(projected_axis_points[1]), font, 0.5, (0, 0, 255), 2)
            cv2.putText(img3, "Y", tuple(projected_axis_points[2]), font, 0.5, (0, 255, 0), 2)
            cv2.putText(img3, "Z", tuple(projected_axis_points[3]), font, 0.5, (255, 0, 0), 2)

            # 绘制检测到的矩形
            cv2.drawContours(img3, [image_points.astype(int)], -1, (0, 255, 255), 2)

            # 显示位置信息和重投影误差
            cv2.putText(img3, f"Position: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f})",
                        (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(img3, f"Repro Error: {error:.2f} px",
                        (10, 60), font, 0.5, (255, 255, 255), 1)
            cv2.putText(img3, "Using temporal coherence",
                        (10, 90), font, 0.5, (0, 255, 255), 1)

    cv2.imshow("Threshold", frame_thresh)
    cv2.imshow("All Contours", img1)
    cv2.imshow("Filtered Contours", img2)
    cv2.imshow("PnP Result", img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()