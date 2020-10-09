from config import TOK_PATH, CLASSES, TRAIN_PATH
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import string
import enchant
import os
import re
import pandas as pd
# from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def get_features(text_series, maxlen):
    with open(TOK_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequences = tokenizer.texts_to_sequences(text_series)
    return pad_sequences(sequences, maxlen=maxlen)


def clean(abstracts_tmp, labels_tmp):
    f = open('data/stopwords_en.txt', 'r', encoding='utf-8')
    stopwords = [words.replace('\n', '') for words in f.readlines()]
    f.close()
    reg = re.compile(r'<[^>]+>', re.S)
    max_seq_len = 0
    all_words = []
    dic_en = enchant.Dict("en_US")
    t = str.maketrans({key: None for key in string.punctuation})
    abstracts = []
    for i in tqdm(range(len(abstracts_tmp)), desc='CLEANING...'):
        line = abstracts_tmp[i]
        desc = reg.sub('', line)
        desc = re.sub(r'\W+', ' ', desc)
        desc = desc.translate(t).lower()
        desc = desc.split(' ')
        x = ''
        for word in desc:
            if x.find(word) == -1 and word not in stopwords and word != '' and word != ' ' and re.search('\d', word) is None and dic_en.check(word):
                if word not in all_words:
                    all_words.append(word)
                x += word+' '
        x = x.strip()
        if len(x) < 4:
            labels_tmp.drop([i], inplace=True)
            continue
        if len(x.split(' ')) > max_seq_len:
            max_seq_len = len(x)
        abstracts.append(x)
    print('*'*20+'DATA DETAIL'+'*'*20)
    print('MAX SEQ LEN:{}'.format(max_seq_len))
    print('ALL WORDS:{}'.format(len(all_words)))
    return abstracts, labels_tmp


def load_data(path):
    if os.path.basename(path).find('.csv') != - -1:
        datas = pd.read_csv(path)
        labels_tmp = datas[[class_name for class_name in CLASSES]]
        abstracts_tmp = datas.ABSTRACT.tolist()
    elif os.path.basename(path).find('.txt') != -1:
        f = open(path, 'r', encoding='utf-8')
        lines = f.readlines()
        f.close()
        abstracts_tmp = []
        labels_tmp = []
        for line in tqdm(lines, desc='LOAD DATA'):
            items = str(line).split('\t')
            label = str(items[0]).split('|')
            word = re.split(r'\W+', str(items[1]))
            labels_tmp.append(label)
            abstracts_tmp.append(word)
    abstracts, labels = clean(abstracts_tmp, labels_tmp)
    # Imbalanced-learn currently supports binary, multiclass and binarized encoded multiclasss targets. Multilabel and multioutput targets are not supported.
    # smo = SMOTE(random_state=42)
    # abstracts, labels = smo.fit_sample(abstracts, labels)
    print('DATA LEN:{}'.format(len(abstracts)))
    print(labels.sum(axis=0))
    return abstracts, labels


if __name__ == '__main__':
    load_data(TRAIN_PATH)
