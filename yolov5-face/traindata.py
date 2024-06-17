import dlib
import glob
import numpy
import os
import csv
from skimage import io

# 人脸关键点检测器
predictor_path = "dat/shape_predictor_68_face_landmarks.dat"
# 人脸识别模型、提取特征值
face_rec_model_path = "dat/dlib_face_recognition_resnet_model_v1.dat"
# 训练图像文件夹
faces_folder_path = 'pic'

# 加载模型
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

candidate = []  # 存放训练集人物名字
descriptors = []  # 存放训练集人物特征列表

for f in glob.glob(os.path.join(faces_folder_path, "*")):
    print("正在处理: {}".format(f))
    img = io.imread(f)
    candidate.append(f.split('\\')[-1].split('.')[0])
    # 人脸检测
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = sp(img, d)
        # 提取特征
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        v = numpy.array(face_descriptor)
        descriptors.append(v.tolist())

# path为输出路径和文件名，newline=''是为了不出现空行
csvFile = open("face_database.csv", "w+", newline='')
try:
    writer = csv.writer(csvFile)
    # data为list类型
    for i in range(len(candidate)):
        writer.writerow([candidate[i],descriptors[i]])
finally:
    csvFile.close()

print('识别训练完毕！')