from tensorflow.keras.layers import Input, GRU, Dense,BatchNormalization,Dropout
from tensorflow.keras.models import Model


class TEXT_GRU:
    def __init__(self, embedding_layer, class_len, max_sequence_len):
        self.embedding_layer = embedding_layer
        self.class_len = class_len
        self.max_sequence_len = max_sequence_len

    def build(self):
        input = Input(shape=(self.max_sequence_len,))
        embedding_layer = self.embedding_layer(input)
        gru = GRU(
            256,
            kernel_initializer="glorot_uniform",
            recurrent_initializer='normal',
            activation='relu',
        )(embedding_layer)

        batch_normalization = BatchNormalization()(gru)
        dropout = Dropout(0.1)(batch_normalization)
        output = Dense(self.class_len, activation='sigmoid')(dropout)

        model = Model(inputs=input, outputs=output)
        return model

    def build_base(self):
        input = Input(shape=(self.max_sequence_len,))
        embedding_layer = self.embedding_layer(input)

        LSTM_Layer_1 = GRU(128)(embedding_layer)

        output = Dense(self.class_len, activation='sigmoid')(LSTM_Layer_1)
        model = Model(inputs=input, outputs=output)
        return model
