import pandas as pd
import cv2
import os
import numpy as np
from skimage.feature import hog

ONE_HOT_ENCODING = True
GET_HOG_WINDOWS_FEATURES = True
GET_HOG_FEATURES = True


OUTPUT_FOLDER_NAME = "D:\\深度学习项目代码\\file\code\\emotion recognition\\driver_feature_small_fish1_second"
data_path = 'Train_data_list.csv'

data_path_abs = 'C:\\Users\\Administrator\\Downloads\\fish_image\\fish_image'


SELECTED_LABELS = [i for i in range(0,23)]
# SELECTED_LABELS = [0,1,2,3,4,5,6,7,8,9]

new_labels= SELECTED_LABELS


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


img_list_all = os.listdir(data_path_abs)
count=0
for key,v in enumerate(img_list_all):
    print(key)
    for key1,v1 in enumerate(os.listdir(data_path_abs+"/"+v)):
        # print(data_path_abs+"/"+v+"/"+v1)
        input_img = cv2.imread(data_path_abs+"/"+v+"/"+v1)
    #         cv2.imshow('Segmentation', input_img)
    #         cv2.waitKey(1500)
    #         cv2.destroyAllWindows()
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(input_img, (150, 150))
        # print(get_new_label(key, one_hot_encoding=ONE_HOT_ENCODING))
        labels_list_test.append(get_new_label(key, one_hot_encoding=ONE_HOT_ENCODING))
        # cv2.imshow("img", image)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

        img_data_list_test.append(gaussian_blur)

        # cv2.imshow("img", gaussian_blur)
        # cv2.imshow("img", image)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()

        count+=1
    #         break
    print(count)



print(len(img_data_list_test))
print(len(labels_list_test))
print(len(hog_features_test))
save_n = 'test'
np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/images.npy', img_data_list_test)

if ONE_HOT_ENCODING:
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
else:
    np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/labels.npy', labels_list_test)
# np.save(OUTPUT_FOLDER_NAME + '/' + save_n + '/hog_features.npy', hog_features_test)


img_data_list_test = []
labels_list_test = []
hog_features_test = []

