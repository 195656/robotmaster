import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 相机内参矩阵
camera_matrix = np.array([[812.74353412, 0., 999.84368755],
                          [0., 811.58454708, 504.75250109],
                          [0., 0., 1.]])

# 畸变系数
dist_coeffs = np.array([-1.99194847e-02, -2.54440885e-04, -1.41682184e-03,
                        2.51937464e-03, 1.43636328e-05])

# 定义矩形的3D世界坐标点（单位：厘米），中心为原点
rect_length = 16  # 长
rect_width = 10  # 宽
object_points = np.array([
    [-rect_length / 2, -rect_width / 2, 0],  # 左下
    [rect_length / 2, -rect_width / 2, 0],  # 右下
    [rect_length / 2, rect_width / 2, 0],  # 右上
    [-rect_length / 2, rect_width / 2, 0]  # 左上
], dtype=np.float32)

# 坐标轴长度
axis_length = max(rect_length, rect_width) * 1.5
axis_points = np.array([
    [0, 0, 0],  # 原点
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, axis_length]  # Z轴 (垂直于纸面向外)
], dtype=np.float32)

prev_rvec = None
prev_tvec = None
prev_z_axis = None
alpha = 0.7  # 增加平滑系数以减少抖动
z_smooth_alpha = 0.8  # Z轴单独平滑系数


def sort_points_clockwise(points):
    # 计算中心点
    center = np.mean(points, axis=0)
    # 计算每个点相对于中心点的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # 按角度排序
    sorted_indices = np.argsort(angles)
    # 返回排序后的点
    return points[sorted_indices]


def ensure_clockwise_order(points):
    """确保点是顺时针顺序"""
    # 计算多边形的面积
    area = 0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    # 如果面积为负，说明点是逆时针顺序，需要反转
    if area < 0:
        return points[::-1]
    return points


def smooth_z_axis(current_z_axis, prev_z_axis, alpha):
    """对Z轴方向进行平滑处理"""
    if prev_z_axis is None:
        return current_z_axis

    # 计算两个向量之间的角度
    dot_product = np.dot(current_z_axis, prev_z_axis)
    norm_product = np.linalg.norm(current_z_axis) * np.linalg.norm(prev_z_axis)
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))

    # 如果角度变化太大，使用上一帧的Z轴
    if np.degrees(angle) > 30:  # 30度阈值
        return prev_z_axis

    # 否则进行平滑
    return alpha * prev_z_axis + (1 - alpha) * current_z_axis


# 创建窗口并调整大小
cv2.namedWindow("PnP Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PnP Result", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("错误：无法读取帧")
        break

    img1 = frame.copy()
    img2 = frame.copy()
    img3 = frame.copy()  # 用于显示PnP结果的图像

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)

    _, frame_thresh = cv2.threshold(blurred, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)

    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

    # 筛选轮廓
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter  # 增加epsilon值以减少顶点数量
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查是否为四边形
        if len(approx) == 4:
            # 转换为点数组
            points = approx.reshape(4, 2)

            # 对点进行排序（基于极角）
            sorted_points = sort_points_clockwise(points)
            sorted_points = ensure_clockwise_order(sorted_points)

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
            valid_angles = all(70 <= angle <= 110 for angle in angles)

            # 如果角度有效，则处理这个四边形
            if valid_angles:
                filtered_contours.append(np.array(sorted_points, dtype=np.float32))
                cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2)

    # 对每个检测到的矩形进行PnP计算
    for rect_points in filtered_contours:
        # 使用亚像素角点检测提高精度
        rect_points_subpix = rect_points.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rect_points_subpix = cv2.cornerSubPix(frame_gray, rect_points_subpix, (5, 5), (-1, -1), criteria)

        # 使用solvePnP计算姿态
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            rect_points_subpix,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # 应用简单的指数平滑
            if prev_rvec is not None and prev_tvec is not None:
                rvec = alpha * rvec + (1 - alpha) * prev_rvec
                tvec = alpha * tvec + (1 - alpha) * prev_tvec

            # 更新前一帧的值
            prev_rvec = rvec
            prev_tvec = tvec

            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 确保旋转矩阵是右手坐标系
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix = -rotation_matrix
                rvec, _ = cv2.Rodrigues(rotation_matrix)

            # 获取当前Z轴方向
            current_z_axis = rotation_matrix[:, 2]  # 旋转矩阵的第三列是Z轴方向

            # 对Z轴方向进行平滑
            if prev_z_axis is not None:
                smoothed_z_axis = smooth_z_axis(current_z_axis, prev_z_axis, z_smooth_alpha)

                # 更新旋转矩阵的Z轴
                rotation_matrix[:, 2] = smoothed_z_axis

                # 重新正交化旋转矩阵
                U, S, Vt = np.linalg.svd(rotation_matrix)
                rotation_matrix = np.dot(U, Vt)

                # 转换回旋转向量
                rvec, _ = cv2.Rodrigues(rotation_matrix)

            # 更新前一帧的Z轴
            prev_z_axis = rotation_matrix[:, 2]

            # 投影到2D图像平面
            projected_axis_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_axis_points = projected_axis_points.reshape(-1, 2).astype(int)

            # 投影矩形角点
            projected_rect_points, _ = cv2.projectPoints(
                object_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_rect_points = projected_rect_points.reshape(-1, 2).astype(int)

            # 原点坐标
            origin = tuple(projected_axis_points[0])

            # 绘制坐标轴
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
            cv2.drawContours(img3, [rect_points.astype(int)], -1, (0, 255, 255), 2)

            # 绘制投影的矩形（用于验证坐标轴是否正确）
            cv2.drawContours(img3, [projected_rect_points], -1, (255, 255, 0), 2)

            # 显示姿态信息
            text = f"Position: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f})"
            cv2.putText(img3, text, (10, 30), font, 0.7, (255, 255, 255), 2)

            # 计算并显示重投影误差
            reprojected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
            error = cv2.norm(rect_points_subpix, reprojected_points, cv2.NORM_L2) / len(reprojected_points)
            error_text = f"Repro Error: {error:.2f} px"
            cv2.putText(img3, error_text, (10, 60), font, 0.7, (255, 255, 255), 2)

    # 显示结果
    cv2.imshow("Threshold", frame_thresh)
    cv2.imshow("All Contours", img1)
    cv2.imshow("Filtered Contours", img2)
    cv2.imshow("PnP Result", img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()