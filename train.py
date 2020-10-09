# -*- coding: utf-8 -*-
import datetime
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import numpy as np
from model import *
from config import *
from load_data import load_data


def build_matrix(embeddings_index, word_index):
    embedding_matrix = np.zeros((MAX_WORDS_LEN, EMBED_SIZE))
    for word, i in tqdm(word_index.items(),desc='BUILD EMBEDDING'):
        if i >= MAX_WORDS_LEN:
            continue
        try:
            # word对应的vector
            embedding_vector = embeddings_index[word]
        except:
            # word不存在则使用unknown的vector
            embedding_vector = embeddings_index["unknown"]
        if embedding_vector is not None:
            # 保证embedding_matrix行的向量与word_index中序号一致
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def train(model_type=''):
    abstract, labels = load_data(TRAIN_PATH)
    # 词向量
    tokenizer = Tokenizer(num_words=MAX_WORDS_LEN, lower=True)
    tokenizer.fit_on_texts(abstract)
    sequences = tokenizer.texts_to_sequences(abstract)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LEN)

    with open(TOK_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # glove嵌入
    embeddings_index = dict(get_coefs(*o.split(" "))
                            for o in open(EMBEDDING_PATH))
    glove_embedding_matrix = build_matrix(
        embeddings_index, tokenizer.word_index)

    x_train, x_validation, y_train, y_validation = train_test_split(
        data.tolist(), labels.values.tolist(), test_size=0.1, random_state=123)

    # 获得model
    model = get_model(model_type, glove_embedding_matrix)

    log_dir = LOG_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # ReduceLROnPlateau=当评价指标不在提升时，减少学习率;EarlyStopping=3轮没有进步时,停止;ModelCheckpoint=只保存最好的模型
    callbacks = [
        ReduceLROnPlateau(monitor='categorical_accuracy'),
        EarlyStopping(patience=30, monitor='val_categorical_accuracy'),
        ModelCheckpoint(filepath=MODEL_PATH, save_best_only=True),
        tensorboard_callback
    ]

    history = model.fit(x_train, y_train,
                        epochs=100,
                        batch_size=1024,
                        verbose=1,
                        validation_data=(x_validation, y_validation),
                        callbacks=callbacks)
    val_cat_acc = history.history['val_categorical_accuracy']
    best_score = max(val_cat_acc)
    best_epoch = val_cat_acc.index(best_score)
    return str(model_type).upper(), '%.4f' % best_score, best_epoch


if __name__ == '__main__':
    train(TEXT_GRU)
