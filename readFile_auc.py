import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import numpy as np
import rembg

ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True
from rembg import remove
from PIL import Image

OUTPUT_FOLDER_NAME = "D:\\深度学习项目代码\\file\\second\\driver_feature_small_Sobel_3_224"
data_path = 'D:\\dataset\\auc.distracted.driver.dataset_v2\\v1_cam1_no_split2\\'

data_path_abs = 'D:\\dataset\\auc.distracted.driver.dataset_v2\\v1_cam1_no_split2\\'


original_labels = [0,1,2,3,4,5,6,7,8,9]
SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]

new_labels= SELECTED_LABELS
print(new_labels)


def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)


img_data_list_train=[]
img_data_list_test=[]

labels_list_train = []
labels_list_test = []

landmarks_train = []
landmarks_test = []

hog_features_train = []
hog_features_test = []

hog_images_train = []
hog_images_test = []

# for path in ['Test_data_list.csv','Train_data_list.csv']:
from sklearn.svm import SVC




for path in ['Test_data_list.csv']:



    path = data_path+path
    print(path)
    train = pd.read_csv(path)

    train_data = train['Image']
    train_label = train['Label']

    count = 0
    for k, i in enumerate(train_data):
        if k%100==0:
            print(k, i)

        name = i.split("/")[-1]
        class_data = i.split("/")[-2]
        # print(class_data, name, train_label[k])

        img_list_all = os.listdir(data_path_abs + '/' + class_data)
        # print(img_list_all)

        for d in img_list_all:
            if name == d:
                # print(data_path + class_data+"\\"+name)
                input_img1 = cv2.imread(data_path + class_data + "\\" + name)
                input_img1 = cv2.resize(input_img1, (224, 224))

                labels_list_test.append(get_new_label(train_label[k], one_hot_encoding=ONE_HOT_ENCODING))

                gray = cv2.cvtColor(input_img1, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

                # 轮廓检测
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # 绘制轮廓
                result = input_img1.copy()
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)


                cv2.imshow("img", result)
                cv2.waitKey(2000)
                cv2.destroyAllWindows()
                img_data_list_test.append(input_img1)
                # cv2.imshow('jijia_xy_Scharr_full', input_img1)
                # cv2.imshow('jijia_xy_Scharr_full', sobel_image)
                # plt.imsave("demo1.jpg", input_img1)  # 将改变后的图像保存
                # plt.imsave("demo1.eps", input_img1)  # 将改变后的图像保存
                # plt.imsave("demo1.svg", input_img1)  # 将改变后的图像保存

                # plt.imsave("demo.png", sobel_image)
                # plt.imsave("demo.svg", sobel_image)
                # plt.imsave("demo.eps", sobel_image)
                count+=1
                break
    print(count)
    # if path.split('\\')[-1] == 'Train_data_list.csv':
    #     save_n = 'train'
    # else:
    #     save_n = 'test'
    save_n = 'test'
    # np.random.seed(2024)
    # np.random.shuffle(img_data_list_test)
    # np.random.seed(2024)
    # np.random.shuffle(labels_list_test)
    # np.random.seed(2019)
    # np.random.shuffle(hog_features_test)

    print(len(img_data_list_test))
    print(len(labels_list_test))
    # print(len(hog_features_test))

    np.random.seed(2025)
    np.random.shuffle(img_data_list_test)
    np.random.seed(2025)
    np.random.shuffle(labels_list_test)


    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images.npy', img_data_list_test)

    if ONE_HOT_ENCODING:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    else:
        np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
    # np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/hog_features.npy', hog_features_test)


    img_data_list_test = []
    labels_list_test = []
    # hog_features_test = []

