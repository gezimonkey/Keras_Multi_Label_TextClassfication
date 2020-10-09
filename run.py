from train import *
from predict import *
from config import *


def train_and_predict():
    result=[]
    models=[CNN,FAST,CHAR_CNN,TEXT_ATT_BI_GRU,TEXT_ATT_BI_LSTM,TEXT_BI_GRU,TEXT_BI_LSTM,TEXT_GRU,TEXT_LSTM]
    # models=[CNN,FAST]
    for model in models:
        mode_type,best_socre,best_epoch=train(model)
        result.append([mode_type,best_socre,best_epoch])
        predict(model)
    print('| 分类器(Classifier) | val_categorical_accuracy | epochs |')
    print('| :----------------- | :----------------------- | :----- |')
    for r in result:
        print('| {}                | {}                   | {}     |'.format(r[0],r[1],r[2]))
    

if __name__ == '__main__':
    train_and_predict()
