########################
# 识别人脸图像
########################

import cv2
import os
from face_train.face_train import Model


if __name__ == '__main__':
    path_name = '../images-data/test-01/6573530'

    #加载模型
    model = Model()
    model.load_model(file_path='../model/aggregate.face.model.h5')

    # 框住人脸的矩形边框颜色
    color = (0, 255, 0)

    # 人脸识别分类器本地存储路径
    # Windows:
    # cascade_path = "C:\Program Files (x86)\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"
    # Linux:
    cascade_path = "/home/alecshan/.conda/pkgs/opencv-4.1.0-py37h3aa1047_6/share/" \
                   "opencv4/haarcascades/haarcascade_frontalface_alt2.xml"

    # 循环检测识别人脸图像
    for dir_item in os.listdir(path_name):
        full_path = path_name + '/' + dir_item

        src_image = cv2.imread(full_path)

        # 图像灰化，降低计算复杂度
        src_image_gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)

        # 使用人脸识别分类器，读入分类器
        cascade = cv2.CascadeClassifier(cascade_path)

        # 利用分类器识别出哪个区域为人脸
        faceRects = cascade.detectMultiScale(src_image_gray, scaleFactor=1.2, minNeighbors=2, minSize=(32, 32))
        if len(faceRects) > 0:
            for faceRect in faceRects:
                x, y, w, h = faceRect

                # 截取脸部图像提交给模型识别这是谁
                image = src_image[y: y + h, x: x + w]       #(改)
                faceID = model.face_predict(image)

                cv2.rectangle(src_image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness=2)
                #face_id判断（改）
                for i in range(len(os.listdir('../images-data/test-01/'))):
                    if i == faceID:
                        # 文字提示是谁
                        cv2.putText(src_image, os.listdir('../images-data/test-01/')[i],
                                    (x + 30, y + 30),  # 坐标
                                    cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                                    1,  # 字号
                                    (255, 0, 255),  # 颜色
                                    2)  # 字的线宽

        # show image
        cv2.imshow("AI recognize image", src_image)

        # save image
        new_file_path = path_name + '/new_' + dir_item
        cv2.imwrite(new_file_path, src_image)

        # 等待10毫秒看是否有按键输入
        k = cv2.waitKey(10)
        # 如果输入q则退出循环
        if k & 0xFF == ord('q'):
            break

    # 销毁所有窗口
    cv2.destroyAllWindows()
