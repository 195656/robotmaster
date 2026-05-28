import cv2
import numpy as np
import glob


def check_calibration_accuracy(camera_matrix, dist_coeffs, calibration_images_path, pattern_size, square_size):
    """
    通过重投影误差验证相机内参的准确性
    """
    # 准备物体点 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储物体点和图像点
    objpoints = []  # 3D点在世界坐标系中
    imgpoints = []  # 2D点在图像平面中

    # 获取校准图像
    images = glob.glob(calibration_images_path)

    total_error = 0
    valid_images = 0

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            valid_images += 1
            objpoints.append(objp)

            # 亚像素精确化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # 计算重投影误差
            _, rvecs, tvecs = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            imgpoints2, _ = cv2.projectPoints(objp, rvecs, tvecs, camera_matrix, dist_coeffs)
            error = cv2.norm(corners2, imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

            print(f"图像 {fname}: 重投影误差 = {error:.3f} 像素")

    if valid_images > 0:
        mean_error = total_error / valid_images
        print(f"\n平均重投影误差: {mean_error:.3f} 像素")
        print(f"验证图像数量: {valid_images}")

        # 评估标准
        if mean_error < 0.5:
            print("✅ 内参非常准确")
        elif mean_error < 1.0:
            print("✅ 内参准确")
        elif mean_error < 2.0:
            print("⚠️  内参一般，建议重新校准")
        else:
            print("❌ 内参不准确，必须重新校准")

        return mean_error
    else:
        print("未找到有效的校准图像")
        return None


# 使用示例
camera_matrix = np.array([[812.74353412, 0., 999.84368755],
                          [0., 811.58454708, 504.75250109],
                          [0., 0., 1.]])

dist_coeffs = np.array([0.00197989, -0.01936813, -0.00234692,
                        0.00864558, -0.00133246])

# 检查内参准确性
mean_error = check_calibration_accuracy(
    camera_matrix,
    dist_coeffs,
    r"C:\Users\25406\Desktop\biaoding/*.jpg",  # 校准图像路径
    (11, 8),  # 棋盘格内角点数量
    1  # 方格大小（厘米）
)