import cv2
import numpy as np

# 相机内参矩阵
camera_matrix = np.array([[812.74353412, 0., 999.84368755],
                          [0., 811.58454708, 504.75250109],
                          [0., 0., 1.]])

# 畸变系数
dist_coeffs = np.array([0.00197989, -0.01936813, -0.00234692,
                        0.00864558, -0.00133246])

# 定义矩形的3D世界坐标点（单位：厘米），中心原点
rect_length = 16  # 长
rect_width = 10  # 宽
object_points = np.array([
    [-rect_length / 2, -rect_width / 2, 0],  # 左下
    [rect_length / 2, -rect_width / 2, 0],  # 右下
    [rect_length / 2, rect_width / 2, 0],  # 右上
    [-rect_length / 2, rect_width / 2, 0],  # 左上
], dtype=np.float32)

# 坐标轴长度
axis_length = max(rect_length, rect_width) * 0.8
axis_points = np.array([
    [0, 0, 0],
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, -axis_length]  # Z轴 (垂直于纸面向内)
], dtype=np.float32)


def sort_points_clockwise(points):
    """按顺时针方向排序点"""
    center = np.mean(points, axis=0)
    # 计算每个点相对于中心点的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    # 按角度排序（顺时针）
    return points[np.argsort(-angles)]


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


def is_valid_rectangle(approx):
    """检查轮廓是否为有效的矩形"""
    if len(approx) != 4:
        return False

    # 转换为点数组
    points = approx.reshape(4, 2)

    # 计算边长
    side_lengths = []
    for i in range(4):
        pt1 = points[i]
        pt2 = points[(i + 1) % 4]
        side_lengths.append(np.linalg.norm(pt1 - pt2))

    # 检查边长比例
    max_side = max(side_lengths)
    min_side = min(side_lengths)
    if max_side / min_side > 2.0:  # 允许一定的比例变化
        return False

    # 检查角度
    angles = []
    for i in range(4):
        pt1 = points[i]
        pt2 = points[(i + 1) % 4]
        pt3 = points[(i + 2) % 4]

        vec1 = pt1 - pt2
        vec2 = pt3 - pt2

        angle_rad = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        angle_deg = np.abs(np.degrees(angle_rad)) % 180

        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        angles.append(angle_deg)

    # 检查角度是否接近90度（允许±25度的误差）
    valid_angles = all(65 <= angle <= 115 for angle in angles)

    return valid_angles


def rotation_matrix_to_euler_angles(rotation_matrix):
    """将旋转矩阵转换为欧拉角"""
    sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    # 转换为角度
    x = np.degrees(x)
    y = np.degrees(y)
    z = np.degrees(z)

    return x, y, z


def calculate_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """计算重投影误差"""
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points, projected_points, cv2.NORM_L2) / len(projected_points)
    return error, projected_points


def process_frame(frame):
    """处理单帧图像并返回结果"""

    # 创建处理后的图像副本
    result_frame = frame.copy()

    # 转换为灰度图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 预处理
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    _, frame_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # 增加最小面积要求
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.01 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查是否为有效的矩形
        if is_valid_rectangle(approx):
            # 转换为点数组并排序
            points = approx.reshape(4, 2)
            sorted_points = sort_points_clockwise(points)
            sorted_points = ensure_clockwise_order(sorted_points)
            filtered_contours.append(np.array(sorted_points, dtype=np.float32))

    # 对每个检测到的矩形进行PnP计算
    for rect_points in filtered_contours:
        # 使用亚像素角点检测提高精度
        rect_points_subpix = rect_points.reshape(-1, 1, 2).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        rect_points_subpix = cv2.cornerSubPix(frame_gray, rect_points_subpix, (5, 5), (-1, -1), criteria)

        # 尝试不同的PnP算法
        methods = [
            cv2.SOLVEPNP_ITERATIVE,
            cv2.SOLVEPNP_EPNP,
            cv2.SOLVEPNP_P3P,
            cv2.SOLVEPNP_AP3P
        ]

        best_error = float('inf')
        best_rvec = None
        best_tvec = None
        best_method = None

        for method in methods:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points,
                    rect_points_subpix,
                    camera_matrix,
                    dist_coeffs,
                    flags=method
                )

                if success:
                    # 计算重投影误差
                    error, _ = calculate_reprojection_error(object_points, rect_points_subpix, rvec, tvec,
                                                            camera_matrix, dist_coeffs)

                    if error < best_error:
                        best_error = error
                        best_rvec = rvec
                        best_tvec = tvec
                        best_method = method
            except:
                continue

        # 使用最佳结果
        if best_rvec is not None:
            rvec, tvec = best_rvec, best_tvec

            # 将旋转向量转换为旋转矩阵
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # 确保旋转矩阵是右手坐标系
            if np.linalg.det(rotation_matrix) < 0:
                rotation_matrix = -rotation_matrix
                rvec, _ = cv2.Rodrigues(rotation_matrix)

            # 计算重投影误差
            error, projected_points = calculate_reprojection_error(object_points, rect_points_subpix, rvec, tvec,
                                                                   camera_matrix, dist_coeffs)

            # 投影到2D图像平面
            projected_axis_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            projected_axis_points = projected_axis_points.reshape(-1, 2).astype(int)

            # 原点坐标
            origin = tuple(projected_axis_points[0])

            # 绘制坐标轴
            # X轴 (红色)
            cv2.line(result_frame, origin, tuple(projected_axis_points[1]), (0, 0, 255), 3)
            # Y轴 (绿色)
            cv2.line(result_frame, origin, tuple(projected_axis_points[2]), (0, 255, 0), 3)
            # Z轴 (蓝色)
            cv2.line(result_frame, origin, tuple(projected_axis_points[3]), (255, 0, 0), 3)

            # 添加坐标轴标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result_frame, "X", tuple(projected_axis_points[1]), font, 0.5, (0, 0, 255), 2)
            cv2.putText(result_frame, "Y", tuple(projected_axis_points[2]), font, 0.5, (0, 255, 0), 2)
            cv2.putText(result_frame, "Z", tuple(projected_axis_points[3]), font, 0.5, (255, 0, 0), 2)

            # 显示姿态信息
            text_pos = f"Position: ({tvec[0][0]:.1f}, {tvec[1][0]:.1f}, {tvec[2][0]:.1f})"
            cv2.putText(result_frame, text_pos, (10, 30), font, 0.7, (255, 255, 255), 2)

            # 显示重投影误差
            error_text = f"Reprojection Error: {error:.2f} px"
            cv2.putText(result_frame, error_text, (10, 60), font, 0.7, (255, 255, 255), 2)

            # 计算欧拉角
            x, y, z = rotation_matrix_to_euler_angles(rotation_matrix)

            text_rot = f"Rotation: ({x:.1f}, {y:.1f}, {z:.1f})"
            cv2.putText(result_frame, text_rot, (10, 90), font, 0.7, (255, 255, 255), 2)

    return result_frame, frame_thresh


def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 0表示默认摄像头

    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return

    print("按 'q' 键退出程序")
    print("按 's' 键保存当前帧")

    frame_count = 0
    save_count = 0

    while True:
        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取帧")
            break

        # 处理帧
        result_frame, threshold_frame = process_frame(frame)

        # 显示结果
        cv2.imshow("Original", frame)
        cv2.imshow("Threshold", threshold_frame)
        cv2.imshow("PnP Result", result_frame)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            cv2.imwrite(f"frame_{save_count:03d}.jpg", frame)
            cv2.imwrite(f"result_{save_count:03d}.jpg", result_frame)
            print(f"已保存帧 {save_count}")
            save_count += 1

        frame_count += 1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()