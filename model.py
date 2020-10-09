from tensorflow.keras.layers import Embedding
from config import *
from models.char_cnn import *
from models.cnn import *
from models.fast import *
from models.text_att_bi_gru import *
from models.text_att_bi_lstm import *
from models.text_bi_gru import *
from models.text_bi_lstm import *
from models.text_gru import *
from models.text_lstm import *


def create_embedding(embedding_matrix):
    return Embedding(input_dim=MAX_WORDS_LEN,
                     output_dim=EMBED_SIZE,
                     weights=[embedding_matrix],
                     input_length=MAX_SEQUENCE_LEN,
                     trainable=False)
                     
def get_model(model_type, embedding_matrix):
    class_len = len(CLASSES)
    model_class=globals()[model_type](create_embedding(embedding_matrix), class_len,MAX_SEQUENCE_LEN)
    model=model_class.build()
    # 优化器=adam 损失函数=二分类交叉熵损失函数 评价指标=输出所有结果的概率
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['categorical_accuracy'])
    return model