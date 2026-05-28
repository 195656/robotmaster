import cv2
import numpy as np
import os
import glob

# 设置图片文件夹路径
folder_path = r"C:\Users\25406\Desktop\biaoding"

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
axis_length = max(rect_length, rect_width) * 1.5
axis_points = np.array([
    [0, 0, 0],
    [axis_length, 0, 0],  # X轴
    [0, axis_length, 0],  # Y轴
    [0, 0, -axis_length]  # Z轴 (垂直于纸面向内，注意这里是负值)
], dtype=np.float32)


def sort_points_clockwise(points):
    """按顺时针方向排序点"""
    # 计算中心点
    center = np.mean(points, axis=0)

    # 计算每个点相对于中心点的角度
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

    # 按角度排序（顺时针）
    sorted_indices = np.argsort(-angles)

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


def calculate_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix, dist_coeffs):
    """计算重投影误差"""
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points, projected_points, cv2.NORM_L2) / len(projected_points)
    return error, projected_points


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


def process_image(frame, image_name):
    """处理单张图像并返回结果图像和PnP数据"""
    # 创建处理后的图像副本
    img1 = frame.copy()  # 所有轮廓
    img2 = frame.copy()  # 筛选后的轮廓
    img3 = frame.copy()  # PnP结果

    # 转换为灰度图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 预处理
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    _, frame_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    frame_thresh = cv2.morphologyEx(frame_thresh, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, hierarchy = cv2.findContours(frame_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img1, contours, -1, (0, 255, 0), 2)

    # 筛选轮廓
    filtered_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 300:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查是否为四边形
        if len(approx) == 4:
            # 转换为点数组
            points = approx.reshape(4, 2).astype(np.float32)

            # 对点进行排序（基于极角）
            sorted_points = sort_points_clockwise(points)

            # 确保点是顺时针顺序
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
                filtered_contours.append(sorted_points)
                cv2.drawContours(img2, [approx], -1, (0, 255, 0), 2)

    # 对每个检测到的矩形进行PnP计算
    pnp_results = []
    for i, rect_points in enumerate(filtered_contours):
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

            # 显示重投影误差
            error_text = f"Reprojection Error: {error:.2f} px"
            cv2.putText(img3, error_text, (10, 60), font, 0.7, (255, 255, 255), 2)

            # 计算欧拉角
            x, y, z = rotation_matrix_to_euler_angles(rotation_matrix)

            # 保存结果
            pnp_results.append({
                'error': error,
                'rotation_matrix': rotation_matrix,
                'euler_angles': (x, y, z),
                'translation_vector': tvec.flatten(),
                'method': best_method
            })

            # 打印结果
            print(f"图像: {image_name}, 矩形 {i + 1} 的重投影误差: {error} 像素")
            print(f"使用的PnP方法: {best_method}")
            print(f"欧拉角 (XYZ): ({x:.2f}, {y:.2f}, {z:.2f}) 度")

    return img1, img2, img3, frame_thresh, pnp_results


# 获取文件夹中的所有图片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
image_paths = []
for extension in image_extensions:
    image_paths.extend(glob.glob(os.path.join(folder_path, extension)))

# 按文件名排序
image_paths.sort()

print(f"找到 {len(image_paths)} 张图片")

# 创建窗口
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("All Contours", cv2.WINDOW_NORMAL)
cv2.namedWindow("Filtered Contours", cv2.WINDOW_NORMAL)
cv2.namedWindow("PnP Result", cv2.WINDOW_NORMAL)

# 处理每张图片
for i, image_path in enumerate(image_paths):
    print(f"\n处理第 {i + 1}/{len(image_paths)} 张图片: {os.path.basename(image_path)}")

    # 读取图像
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误：无法加载图像 {image_path}")
        continue

    # 处理图像
    img1, img2, img3, frame_thresh, pnp_results = process_image(frame, os.path.basename(image_path))

    # 显示结果
    cv2.imshow("Threshold", frame_thresh)
    cv2.imshow("All Contours", img1)
    cv2.imshow("Filtered Contours", img2)
    cv2.imshow("PnP Result", img3)

    # 显示图像文件名
    cv2.setWindowTitle("PnP Result", f"PnP Result - {os.path.basename(image_path)}")

    # 等待用户按键
    print("按任意键继续下一张图片，按 'q' 退出...")
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        break

# 释放资源
cv2.destroyAllWindows()
print("处理完成")