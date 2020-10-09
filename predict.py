from config import *
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import re
import os
from load_data import *


def predict(model_type=''):
    x, y = load_data(PREDICT_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
    predict_results = {}
    for i in tqdm(range(len(x)), desc='PREDICTING...'):
        seg = x[i]
        result = model.predict(get_features([seg], MAX_SEQUENCE_LEN)).mean(0)
        id_top = list(list(np.where(result > PREDICT_LEVEL))[0])
        id_max = list(result).index(max(result))
        if id_max not in id_top:
            id_top.append(id_max)
        for id in id_top:
            predict_results[id] = predict_results[id] + \
                1 if id in predict_results else 1
    print('分类结果({})'.format(model_type.upper()))
    print('| 分类(Class)               | 数量(Quantity) |')
    print('| :----------------- | :--- |')
    for i in range(len(CLASSES)):
        value = predict_results[i] if i in predict_results.keys() else '0'
        print('|  {}    |    {}    |'.format(str(CLASSES[i]), str(value)))


if __name__ == '__main__':
    predict(TEXT_GRU)
