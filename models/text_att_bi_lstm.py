from tensorflow.keras.layers import Input,LSTM,Dense,Bidirectional
from tensorflow.keras.models import Model
from models.attention import Attention
class TEXT_ATT_BI_LSTM:
    def __init__(self, embedding_layer, class_len, max_sequence_len):
        self.embedding_layer = embedding_layer
        self.class_len = class_len
        self.max_sequence_len = max_sequence_len

    def build(self):
        input = Input(shape=(self.max_sequence_len,))
        embedding_layer = self.embedding_layer(input)

        bi = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
        att = Attention()(bi)

        output = Dense(self.class_len, activation='sigmoid')(att)
        model = Model(inputs=input, outputs=output)
        return model