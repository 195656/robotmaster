import numpy as np
import cv2
import glob
import os


def calibrate_camera(folder_path, chessboard_size, square_size, image_extensions=['.jpg', '.jpeg', '.png']):
    """
    相机标定函数

    参数:
    folder_path: 包含标定图片的文件夹路径
    chessboard_size: 棋盘格内角点数量 (width, height)
    square_size: 棋盘格每个方格的实际尺寸（单位：米）
    image_extensions: 支持的图片格式列表

    返回:
    camera_matrix: 相机内参矩阵
    dist_coeffs: 畸变系数
    rvecs: 旋转向量
    tvecs: 平移向量
    """

    # 定义棋盘格角点的世界坐标
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 存储世界坐标和图像坐标
    objpoints = []  # 世界坐标系中的3D点
    imgpoints = []  # 图像坐标系中的2D点

    # 获取文件夹中所有指定格式的图片
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        images.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))

    if not images:
        print(f"在文件夹 '{folder_path}' 中未找到标定图片，支持的格式: {image_extensions}")
        return None, None, None, None

    print(f"找到 {len(images)} 张图片")

    # 成功检测到角点的图片数量
    success_count = 0

    # 遍历所有图片
    for i, fname in enumerate(images):
        print(f"处理图片 {i + 1}/{len(images)}: {os.path.basename(fname)}")
        img = cv2.imread(fname)
        if img is None:
            print(f"  无法读取图片: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 查找棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # 如果找到，添加对象点和图像点
        if ret:
            objpoints.append(objp)

            # 提高角点检测精度
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            success_count += 1
            print(f"  成功检测到角点")

            # 绘制并显示角点
            img_with_corners = cv2.drawChessboardCorners(img.copy(), chessboard_size, corners2, ret)
            # 调整显示窗口大小
            display_img = cv2.resize(img_with_corners, (960, 540))
            cv2.imshow('Chessboard Corners', display_img)
            cv2.waitKey(300)  # 显示300毫秒
        else:
            print(f"  未找到棋盘格角点")

    cv2.destroyAllWindows()

    print(f"\n成功在 {success_count} 张图片中检测到角点")

    if success_count < 5:
        print("检测到角点的图片数量不足，至少需要5张图片才能进行标定")
        return None, None, None, None

    # 进行相机标定
    print("开始相机标定...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    # 计算重投影误差
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print(f"\n标定完成!")
    print(f"重投影误差: {mean_error / len(objpoints):.5f} (越小越好，通常应小于0.1)")
    print(f"相机内参矩阵:\n{camera_matrix}")
    print(f"畸变系数: {dist_coeffs.ravel()}")

    return camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration_results(filename, camera_matrix, dist_coeffs):
    """保存标定结果到文件"""
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"标定结果已保存到 {filename}")


def load_calibration_results(filename):
    """从文件加载标定结果"""
    data = np.load(filename)
    return data['camera_matrix'], data['dist_coeffs']


def undistort_image(image, camera_matrix, dist_coeffs):
    """使用标定结果校正图像畸变"""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 校正图像
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # 裁剪图像
    x, y, w, h = roi
    undistorted = undistorted[y:y + h, x:x + w]

    return undistorted


# 使用示例
if __name__ == "__main__":
    # 设置参数
    folder_path = r"C:\Users\25406\Desktop\biaoding"  # 标定图片文件夹路径
    chessboard_size = (11, 8)  # 棋盘格内角点数量 (width, height)
    square_size = 0.01  # 每个棋盘格方格的尺寸（单位：米）

    # 进行相机标定
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        folder_path, chessboard_size, square_size)

    if camera_matrix is not None:
        # 保存标定结果
        save_calibration_results("camera_calibration.npz", camera_matrix, dist_coeffs)

        # 测试校正效果 - 使用第一张成功检测的图片
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
        if images:
            test_image = cv2.imread(images[0])
            if test_image is not None:
                undistorted = undistort_image(test_image, camera_matrix, dist_coeffs)

                # 并排显示原图和校正后的图像
                combined = np.hstack((test_image, undistorted))
                # 调整显示大小
                display_img = cv2.resize(combined, (1280, 480))
                cv2.imshow("Original (左) vs Undistorted (右)", display_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # 保存校正后的图像
                cv2.imwrite("undistorted_example.jpg", undistorted)