import os
import time
import numpy as np

RS = 123        # random state variable

from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA


if __name__ == ' __main__':
    gram_list = ['1gram_768/', '2gram_768/', '3gram_768/', '4gram_768/']
    path = 'D:/virus/image/'

    image_list = [[],[],[],[]]
    label_list = []

    gram_index = 0

    for gram in tqdm(gram_list):
        PATH = path+gram
        image_file_list = os.listdir(PATH)

        for file_name in image_file_list:
            PATH2 = PATH + file_name

            if file_name[-3:]=='npy':
                image = np.load(PATH2).reshape(-1)/255.0

            else:
                image = np.array(Image.open(PATH2)).reshape(-1)/255.0
                label = int(file_name[-5]) -1

            image_list[gram_index].append(image)
            label_list.append(label)

        gram_index += 1

    for i in range(3):
        image_list[i] = np.asarray(image_list[i])

    y = np.asarray(label_list[:10736])

    x_1 = image_list[0]
    x_2 = image_list[1]
    x_3 = image_list[2]
    x_4 = image_list[3]
    
    
    
    
    
    
    
    # 1-gram
    selected_n_gram = x_1
    clf_pca = PCA()
    clf_pca.fit(selected_n_gram)
    data1 = clf_pca.fit_transform(selected_n_gram)

    # 2-gram
    selected_n_gram = x_2
    clf_pca = PCA()
    clf_pca.fit(selected_n_gram)
    data2 = clf_pca.fit_transform(selected_n_gram)

    # 3-gram
    selected_n_gram = x_3
    clf_pca = PCA()
    clf_pca.fit(selected_n_gram)
    data3 = clf_pca.fit_transform(selected_n_gram)

    # 4-gram
    selected_n_gram = x_4
    clf_pca = PCA()
    clf_pca.fit(selected_n_gram)
    data4 = clf_pca.fit_transform(selected_n_gram)
    
    
    # 1gram img data save
    np.save(file = 'D:/virus/image/1gram_768_pca/image_arr.npy', arr = data1)
    np.save(file = 'D:/virus/image/1gram_768_pca/label_arr.npy', arr = y )

    # 2gram img data save
    np.save(file = 'D:/virus/image/2gram_768_pca/image_arr.npy', arr = data2)
    np.save(file = 'D:/virus/image/2gram_768_pca/label_arr.npy', arr = y )

    # 3gram img data save
    np.save(file = 'D:/virus/image/3gram_768_pca/image_arr.npy', arr = data3)
    np.save(file = 'D:/virus/image/3gram_768_pca/label_arr.npy', arr = y )

    # 4gram img data save
    np.save(file = 'D:/virus/image/4gram_768_pca/image_arr.npy', arr = data4)
    np.save(file = 'D:/virus/image/4gram_768_pca/label_arr.npy', arr = y )