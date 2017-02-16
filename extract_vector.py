import sqlite3
from sqlite_wrapper import create_table
from sqlite_wrapper import insert_data
import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
from glob import glob
import time

img_path = "./datasets/dataset_005/579519.jpg"


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(400, 400))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def set_image_to_model(img):
    img_tensor = K.variable(img)
    model = VGG16(input_tensor=img_tensor, weights='imagenet', include_top=False)
    return model


def extract_output(model):
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = outputs_dict["block2_conv1"]
    output = layer_features[0, :, :, :]
    return output


def gram_matrix(output):
    features = K.batch_flatten(K.permute_dimensions(output, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    gram = K.eval(gram)
    return gram


def extract_style_vector(gram):
    style_vector = np.r_[gram[:, 0], gram[gram.shape[0]-1, 1:]]
    style_vector = style_vector / np.linalg.norm(style_vector)
    return style_vector


def get_image_path_list(dataset_num):
    if len(dataset_num) != 3:
        print('Usage: dataset_001 -> 001')
        pass
    img_path_list = glob("./datasets" + "/dataset_" + dataset_num + "/*.jpg")
    return img_path_list

if __name__ == "__main__":

    dimention = 255
    dataset_num = "005"
    table_name = "DataSet_" + dataset_num

    img_path_list = get_image_path_list(dataset_num=dataset_num)
    img_path_num = len(img_path_list)

    columns = "id INTEGER PRIMARY KEY, image_id INTEGER, "
    for i in range(1, dimention + 1):
        feature = "feature" + str(i) + " REAL,"
        columns += feature
    conn = sqlite3.connect('illust_vector.db')
    cur = conn.cursor()
    create_table(conn, cur, table_name, columns[:-1])

    for i, img_path in enumerate(img_path_list):
        model = set_image_to_model(preprocess_image(img_path))
        output = extract_output(model)
        gram = gram_matrix(output)
        style_vector = extract_style_vector(gram)
        image_id = img_path.split("\\")[1][:-4]
        ids = [i+1, image_id]
        data = tuple(ids + list(style_vector))
        print(data)
        insert_data(conn, cur, table_name, data)
        print(img_path.split("\\")[1][:-4])
        print("(" + str(i+1)+'/'+str(img_path_num) + ")")

    conn.close()





