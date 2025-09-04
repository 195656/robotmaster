import cv2
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("错误：无法打开摄像头")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 相机内参矩阵(未标定）
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float32)

# 畸变系数
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# 定义正方形的3D世界坐标点（单位：厘米）
square_size = 10.0  # 假设正方形边长为10cm
object_points = np.array([
    [-square_size / 2, -square_size / 2, 0],  # 左下
    [square_size / 2, -square_size / 2, 0],  # 右下
    [square_size / 2, square_size / 2, 0],  # 右上
    [-square_size / 2, square_size / 2, 0]  # 左上
], dtype=np.float32)


axis_length = square_size * 1.5
axis_points = np.array([
    [0, 0, 0],  # 原点
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, -axis_length]  # Z轴
], dtype=np.float32)


prev_rvec = None
prev_tvec = None
alpha = 0.3  # 平滑系数

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

        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(approx)

        # 四边形
        if len(approx) == 4 and 300 < area:
            # 转换为点数组
            points = approx.reshape(4, 2)

            # 使用更稳定的点排序方法
            # 计算边界矩形
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int8(box)

            # 计算中心点
            center = np.mean(box, axis=0)


            # 对点进行排序
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

            # 检查角度是否接近90度
            angle_threshold = 15
            valid_angles = all(70 <= angle <= 110 for angle in angles)

          
            if valid_angles:
                filtered_contours.append(np.array(sorted_points, dtype=np.float32))
                cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2)

    # 对每个检测到的正方形进行PnP计算
    for square_points in filtered_contours:
        # 使用solvePnP计算姿态
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            square_points,
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

            # 投影到2D图像平面
            projected_axis_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_axis_points = projected_axis_points.reshape(-1, 2).astype(int)

            # 原点坐标
            origin = tuple(projected_axis_points[0])

            # 绘制坐标轴
            # X
            cv2.line(img3, origin, tuple(projected_axis_points[1]), (0, 0, 255), 3)
            # Y
            cv2.line(img3, origin, tuple(projected_axis_points[2]), (0, 255, 0), 3)
            # Z
            cv2.line(img3, origin, tuple(projected_axis_points[3]), (255, 0, 0), 3)

            # 添加坐标轴标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img3, "X", tuple(projected_axis_points[1]), font, 0.5, (0, 0, 255), 2)
            cv2.putText(img3, "Y", tuple(projected_axis_points[2]), font, 0.5, (0, 255, 0), 2)
            cv2.putText(img3, "Z", tuple(projected_axis_points[3]), font, 0.5, (255, 0, 0), 2)



            # 绘制检测到的正方形
            cv2.drawContours(img3, [square_points.astype(int)], -1, (0, 255, 255), 2)

    # 显示结果
    cv2.imshow("Threshold", frame_thresh)
    cv2.imshow("All Contours", img1)
    cv2.imshow("Filtered Contours", img2)
    cv2.imshow("PnP Result", img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
