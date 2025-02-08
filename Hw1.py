import sys
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QWidget, QVBoxLayout, QLabel , QMessageBox, QGraphicsScene, QGraphicsView 
from PyQt5.QtGui import QPainter, QPen, QPixmap , QImage
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QPoint
import sys
import cv2
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchsummary
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torchvision.models import resnet50

app = QtWidgets.QApplication(sys.argv) 
global pic1
filter_type = 0
window = QtWidgets.QMainWindow()
window.setWindowTitle("HW1")
window.setGeometry(300, 100, 1000, 800) 
loadfolder = QtWidgets.QPushButton("Load folder ", window)
loadfolder.move(40, 250)
loadimageL = QtWidgets.QPushButton("Load Image_L ", window)
loadimageL.move(40, 300)
loadimageR = QtWidgets.QPushButton("Load Image_R ", window)
loadimageR.move(40, 350)

# Create Q1
groupBox1= QtWidgets.QGroupBox("1.Calibration ", window)
groupBox1.setFixedSize(250 , 400)
groupBox1.move(180, 30)
pushButton1_1 = QtWidgets.QPushButton("1.1 Find corners", window)
pushButton1_1.resize(180, 40)
pushButton1_1.move(220, 60)
pushButton1_2 = QtWidgets.QPushButton("1.2 Find Intrinsic", window)
pushButton1_2.resize(180, 40)
pushButton1_2.move(220, 125)
groupBox1_3= QtWidgets.QGroupBox("1.3 Find Extrinsic ", window)
groupBox1_3.setFixedSize(180 , 130)
groupBox1_3.move(220, 170)
comboBox = QtWidgets.QComboBox(groupBox1_3)
comboBox.setGeometry(10, 30, 160, 30) 
comboBox.addItems([str(i) for i in range(1, 16)]) 
pushButton1_3 = QtWidgets.QPushButton("1.3 Find Extrinsic", window)
pushButton1_3.resize(180, 40)
pushButton1_3.move(220, 250)
pushButton1_4 = QtWidgets.QPushButton("1.4 Find Distortion", window)
pushButton1_4.resize(180, 40)
pushButton1_4.move(220, 320)
pushButton1_5 = QtWidgets.QPushButton("1.5 Show Result", window)
pushButton1_5.resize(180, 40)
pushButton1_5.move(220, 380)
#label1_1 = QtWidgets.QLabel("There are           coins in the image", window)
#label1_1.resize(180, 40)
#label1_1.move(220, 190)
#label1_2 = QtWidgets.QLabel("", window)
#label1_2.resize(180, 40)
#label1_2.move(275, 190)

#Create Q2
groupBox2= QtWidgets.QGroupBox("2.Augmented Reality", window)
groupBox2.setFixedSize(230 , 400)
groupBox2.move(450, 30)
lineEdit = QtWidgets.QLineEdit(groupBox2)
lineEdit.setGeometry(20, 50, 200, 60)  # 設置文字框的位置和大小
lineEdit.setPlaceholderText("")
pushButton2_1 = QtWidgets.QPushButton("2.1 Show words on board", window)
pushButton2_1.resize(180, 40)
pushButton2_1.move(470, 230)
pushButton2_2= QtWidgets.QPushButton("2.2 Show words vertically", window)
pushButton2_2.resize(180, 40)
pushButton2_2.move(470, 300)

#Create Q3
groupBox3= QtWidgets.QGroupBox("3.Stereo Disparity Map ", window)
groupBox3.setFixedSize(230 , 400)
groupBox3.move(730, 30)
pushButton3_1 = QtWidgets.QPushButton("3.1 Stereo disparity map", window)
pushButton3_1.resize(180, 40)
pushButton3_1.move(750, 230)


#create Q4
groupBox4 = QtWidgets.QGroupBox("4.SIFT", window)
groupBox4.setFixedSize(250 , 320)
groupBox4.move(180, 450)
pushButton4_1 = QtWidgets.QPushButton("Load Image1", window)
pushButton4_1.resize(150, 40)
pushButton4_1.move(220 , 490)
pushButton4_2 = QtWidgets.QPushButton("Load Image2", window)
pushButton4_2.resize(150, 40)
pushButton4_2.move(220, 560)
pushButton4_3 = QtWidgets.QPushButton("4.1 Keypoints", window)
pushButton4_3.resize(150, 40)
pushButton4_3.move(220, 630)
pushButton4_4 = QtWidgets.QPushButton("4.4 Matched Keypoints", window)
pushButton4_4.resize(150, 40)
pushButton4_4.move(220, 700)

def loadimage1_clicked():
    global image1
    image1 = open_image_using_dialog()
    #cv2.imshow("Image", imageL)
pushButton4_1.clicked.connect(loadimage1_clicked)

def loadimage2_clicked():
    global image2
    image2 = open_image_using_dialog()
    #cv2.imshow("Image", imageR)
pushButton4_2.clicked.connect(loadimage2_clicked)

def loadimageL_clicked():
    global imageL
    imageL = open_image_using_dialog()
    #cv2.imshow("Image", imageL)
loadimageL.clicked.connect(loadimageL_clicked)

def loadimageR_clicked():
    global imageR
    imageR = open_image_using_dialog()
    #cv2.imshow("Image", imageR)
loadimageR.clicked.connect(loadimageR_clicked)

def open_image_using_dialog():
    options = QFileDialog.Options()
    options |= QFileDialog.ReadOnly
    image_path, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff);;All Files (*)", options=options)
    print(image_path)
    if image_path:
        image = cv2.imread(image_path)
    else:
        image = None
    return image

def load_folder():
    global image_paths , folder_path
    # 打開資料夾選擇對話框
    folder_path = QtWidgets.QFileDialog.getExistingDirectory(None, "Select Folder")

    if folder_path:
        # 選擇資料夾後，找出資料夾中的所有 bmp 格式影像檔案
        image_paths = sorted(
    [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.bmp')],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

        print("Loaded image paths:")
        for path in image_paths:
            print(path)
        if image_paths:
            print(f"{len(image_paths)} images loaded successfully.")
        else:
            print("No BMP images found in the selected folder.")
    else:
        print("No folder selected.")

# 設定 loadfolder 按鈕觸發 load_folder 函式
loadfolder.clicked.connect(load_folder)

def Findcorners():
    chessboard_size = (11, 8)  # 內部角點的尺寸 (width, high)
    winSize = (5, 5)           # 搜索區域範圍
    zeroZone = (-1, -1)        # 無死區
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)  # OpenCV 建議的終止準則

    if not image_paths:
        print("Please load images using the 'Load folder' button first.")
        return

    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}")
            continue

        # 轉換為灰階
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 找到棋盤格角點
        ret, corners = cv2.findChessboardCorners(grayimg, chessboard_size, None)

        if ret:
            # 提高角點準確度
            corners = cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
            # 在影像上繪製角點
            cv2.drawChessboardCorners(image, chessboard_size, corners, ret)

            # 調整影像大小 (例如縮小至 640x640)
            resized_image = cv2.resize(image, (640, 640))

            # 顯示縮小後的影像和角點
            cv2.imshow(f"Image with Corners - {os.path.basename(image_path)}", resized_image)
        else:
            print(f"Chessboard corners not found in {os.path.basename(image_path)}")
pushButton1_1.clicked.connect(Findcorners)
    
def Find_Intrinsic():
    global image_paths
    if not image_paths:
        print("Please load images first.")
        return

    # 設置棋盤格的內部角點
    object_points = np.zeros((8 * 11, 3), np.float32)  # 8x11 棋盤格
    object_points[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 0.02  # 假設每個方格邊長為 0.02m

    image_points = []  # 存放影像的角點
    for img_path in image_paths:
        grayimg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        ret, corners = cv2.findChessboardCorners(grayimg, (11, 8))
        if ret:
            corners = cv2.cornerSubPix(grayimg, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001))
            image_points.append(corners)

    # 獲取影像尺寸
    h, w = grayimg.shape

    # 計算相機內部參數
    ret, intrinsic, distortion, rvecs, tvecs = cv2.calibrateCamera([object_points] * len(image_points), image_points, (w, h), None, None)

    # 儲存內部參數和畸變係數
    window.intrinsic = intrinsic
    window.distortion = distortion
    window.rvecs = rvecs
    window.tvecs = tvecs

    print("Intrinsic Matrix:")
    print(intrinsic)
pushButton1_2.clicked.connect(Find_Intrinsic)

def Find_Extrinsic():
    if not hasattr(window, 'intrinsic') or not hasattr(window, 'distortion') or not hasattr(window, 'rvecs') or not hasattr(window, 'tvecs'):
        print("Please complete the intrinsic matrix calibration first.")
        return

    # 取得選擇的影像編號
    image_index = int(comboBox.currentText()) - 1  # 編號從 0 開始
    if image_index >= len(window.rvecs) or image_index >= len(window.tvecs):
        print("Invalid image index.")
        return

    # 取得該影像的旋轉向量和平移向量
    rvec = window.rvecs[image_index]
    tvec = window.tvecs[image_index]

    # 將旋轉向量轉換為旋轉矩陣
    rotation_matrix, _ = cv2.Rodrigues(rvec)

    # 合併旋轉矩陣和平移向量形成外部參數矩陣
    extrinsic_matrix = np.hstack((rotation_matrix, tvec))

    # 在控制台中輸出結果
    print(f"Extrinsic Matrix for Image {image_index + 1}:")
    print(extrinsic_matrix)
pushButton1_3.clicked.connect(Find_Extrinsic)

def Find_Distortion():
    if hasattr(window, 'distortion'):
        print("Distortion Coefficients:")
        print(window.distortion)
    else:
        print("Please complete intrinsic matrix calibration first.")
pushButton1_4.clicked.connect(Find_Distortion)

def undistort_images():
    if not hasattr(window, 'intrinsic') or not hasattr(window, 'distortion'):
        print("Please complete intrinsic matrix calibration first.")
        return

    # 取得選擇的影像編號
    image_index = int(comboBox.currentText()) - 1  # 編號從 0 開始
    if image_index < 0 or image_index >= len(image_paths):
        print("Invalid image index.")
        return

    # 讀取選定的原始圖片
    img_path = image_paths[image_index]
    grayimg = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # 去畸變處理
    undistorted_img = cv2.undistort(grayimg, window.intrinsic, window.distortion)

    # 調整圖片大小
    grayimg_resized = cv2.resize(grayimg, (640, 640))
    undistorted_img_resized = cv2.resize(undistorted_img, (640, 640))

    # 合併圖片
    combined_img = np.hstack((grayimg_resized, undistorted_img_resized))
    print(f"Selected image index: {image_index}")
    print(f"Image path: {img_path}")
    # 顯示合併的圖片
    cv2.imshow(f"Distorted and Undistorted Images: {os.path.basename(img_path)}", combined_img)
    cv2.waitKey(0)  # 等待按鍵事件來顯示下一張圖片

    cv2.destroyAllWindows()  # 關閉所有圖片窗口
pushButton1_5.clicked.connect(undistort_images)

def show_words_on_board():
        width = 11
        height = 8

        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

        objp = np.zeros((height * width, 3), np.float32)
        objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * 0.02
            
        objpoints = []
        imgpoints = []
        
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
        for image_file in image_files:
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
            if ret:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                imgpoints.append(corners2)
        
        img_size = (2048, 2048)
        ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, 
            imgpoints, 
            img_size, 
            None, 
            None
        )
        print("Camera calibration completed.")

        input_text =  lineEdit.text().upper()
        if not input_text or len(input_text) > 6:
            print("Please input 1-6 characters!")
            return

        try:
            
            fs = cv2.FileStorage("C:\\Users\\user\\OneDrive\\Desktop\\Dataset_CvDl_Hw1\\Dataset_CvDl_Hw1\\Q2_Image\\Q2_db\\alphabet_db_onboard.txt", cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                print("Failed to open database file.")
                return

            predefined_positions = [
                (8, 5, 0),
                (5, 5, 0),
                (2, 5, 0),
                (8, 2, 0),
                (5, 2, 0),
                (2, 2, 0)
            ]
            
            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
            
            for idx, image_file in enumerate(image_files):
                img_path = os.path.join(folder_path, image_file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read image {image_file}")
                    continue
                
                img_copy = img.copy()

                for i, letter in enumerate(input_text):
                    char_node = fs.getNode(letter)
                    if char_node.empty():
                        print(f"Character {letter} not found in database")
                        continue
                    
                    try:
                        char_points = char_node.mat()
                        if char_points is None:
                            continue
                            
                        char_points = char_points.reshape(-1, 3).astype(np.float64)
                        scale = 0.02
                        char_points *= scale
                        
                        char_points += np.array(predefined_positions[i]) * 0.02
                        
                        points_2d, _ = cv2.projectPoints(
                            char_points,
                            rvecs[idx],
                            tvecs[idx],
                            ins,
                            dist
                        )
                        
                        for j in range(0, len(points_2d)-1, 2):
                            pt1 = tuple(points_2d[j][0].astype(int))
                            pt2 = tuple(points_2d[j+1][0].astype(int))
                            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 3)
                            
                    except Exception as e:
                        print(f"Error processing character {letter}: {str(e)}")
                        continue
                
                try:
                    scale_percent = 40
                    width = int(img_copy.shape[1] * scale_percent / 100)
                    height = int(img_copy.shape[0] * scale_percent / 100)
                    resized_img = cv2.resize(img_copy, (width, height), interpolation=cv2.INTER_AREA)
                    
                    cv2.imshow(f'Image {idx + 1}', resized_img)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f'Image {idx + 1}')
                    
                except Exception as e:
                    print(f"Error displaying image: {str(e)}")
            
        except Exception as e:
            print(f"Error: {str(e)}")
            
        finally:
            if 'fs' in locals():
                fs.release()
pushButton2_1.clicked.connect(show_words_on_board)

def show_words_on_vertical():
    width = 11
    height = 8
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2) * 0.02
        
    objpoints = []
    imgpoints = []
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    for image_file in image_files:
        img_path = os.path.join(folder_path, image_file)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    img_size = (2048, 2048)
    ret, ins, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, 
        imgpoints, 
        img_size, 
        None, 
        None
    )
    print("Camera calibration completed.")
    input_text =  lineEdit.text().upper()
    if not input_text or len(input_text) > 6:
        print("Please input 1-6 characters!")
        return
    try:
        
        fs = cv2.FileStorage("C:\\Users\\user\\OneDrive\\Desktop\\Dataset_CvDl_Hw1\\Dataset_CvDl_Hw1\\Q2_Image\\Q2_db\\alphabet_db_vertical.txt", cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print("Failed to open database file.")
            return
        predefined_positions = [
            (8, 5, 0),
            (5, 5, 0),
            (2, 5, 0),
            (8, 2, 0),
            (5, 2, 0),
            (2, 2, 0)
        ]
        
        image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
        
        for idx, image_file in enumerate(image_files):
            img_path = os.path.join(folder_path, image_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Cannot read image {image_file}")
                continue
            
            img_copy = img.copy()
            for i, letter in enumerate(input_text):
                char_node = fs.getNode(letter)
                if char_node.empty():
                    print(f"Character {letter} not found in database")
                    continue
                
                try:
                    char_points = char_node.mat()
                    if char_points is None:
                        continue
                        
                    char_points = char_points.reshape(-1, 3).astype(np.float64)
                    scale = 0.02
                    char_points *= scale
                    
                    char_points += np.array(predefined_positions[i]) * 0.02
                    
                    points_2d, _ = cv2.projectPoints(
                        char_points,
                        rvecs[idx],
                        tvecs[idx],
                        ins,
                        dist
                    )
                    
                    for j in range(0, len(points_2d)-1, 2):
                        pt1 = tuple(points_2d[j][0].astype(int))
                        pt2 = tuple(points_2d[j+1][0].astype(int))
                        cv2.line(img_copy, pt1, pt2, (0, 0, 255), 3)
                        
                except Exception as e:
                    print(f"Error processing character {letter}: {str(e)}")
                    continue
            
            try:
                scale_percent = 40
                width = int(img_copy.shape[1] * scale_percent / 100)
                height = int(img_copy.shape[0] * scale_percent / 100)
                resized_img = cv2.resize(img_copy, (width, height), interpolation=cv2.INTER_AREA)
                
                cv2.imshow(f'Image {idx + 1}', resized_img)
                cv2.waitKey(0)
                cv2.destroyWindow(f'Image {idx + 1}')
                
            except Exception as e:
                print(f"Error displaying image: {str(e)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
    finally:
        if 'fs' in locals():
            fs.release()
pushButton2_2.clicked.connect(show_words_on_vertical)

def Stereo_disparity_map():
    if imageL is None or imageR is None:
        print("Failed to load one of the images.")
    else:
        # 将图像转换为灰度格式
        grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

        stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)


        # 计算视差图
        disparity = stereo.compute(grayL, grayR)

        # 归一化到 [0, 255] 范围
        disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        disparity_normalized = np.uint8(disparity_normalized)  # 转换为 8 位图像以便显示

        # 调整视差图尺寸，方便展示
        resized_disparity = cv2.resize(disparity_normalized, (640, 480))

        # 显示视差图
        cv2.imshow("Disparity Map (Resized)", resized_disparity)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
pushButton3_1.clicked.connect(Stereo_disparity_map)

def show_keypoints():
    if image1 is None:
        print("Image not loaded. Please load Left.jpg first.")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector and detect keypoints
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints on grayscale image
    keypoints_img = cv2.drawKeypoints(gray, keypoints, None, color=(0, 255, 0))

    # Display the image with keypoints
    resized_image = cv2.resize(keypoints_img, (640, 640))
    cv2.imshow("I1 - Keypoints on Left.jpg", resized_image) 
pushButton4_3.clicked.connect(show_keypoints)

def match_keypoints():
    if image1 is None or image2 is None:
        print("Images not loaded. Please load Left.jpg and Right.jpg first.")
        return

    # Convert images to grayscale
    grayL = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Create SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypointsL, descriptorsL = sift.detectAndCompute(grayL, None)
    keypointsR, descriptorsR = sift.detectAndCompute(grayR, None)

    # Match descriptors using BFMatcher and knnMatch
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptorsL, descriptorsR, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    # Draw matched keypoints
    matched_img = cv2.drawMatchesKnn(grayL, keypointsL, grayR, keypointsR, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched image
    resized_image = cv2.resize(matched_img, (640, 640))
    cv2.imshow("I2 - Matched Keypoints between Left and Right Images", resized_image)
pushButton4_4.clicked.connect(match_keypoints)


window.show()

sys.exit(app.exec_())
