from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential


class FAST:
    def __init__(self, embedding_layer, class_len, max_sequence_len):
        self.embedding_layer = embedding_layer
        self.class_len = class_len
        self.max_sequence_len = max_sequence_len

    def build_base(self):
        model = Sequential()
        model.add(self.embedding_layer)
        model.add(GlobalAveragePooling1D())
        model.add(Dense(self.class_len, activation='sigmoid'))
        return model

    def build(self):
        input = Input(shape=(self.max_sequence_len,),
                      dtype='int32', name='input')
        embedded_sequence = self.embedding_layer(input)

        pool_max = GlobalMaxPooling1D()(embedded_sequence)
        pool_ave = GlobalAveragePooling1D()(embedded_sequence)
        x = Concatenate()([pool_ave, pool_max])
        x = Dense(128, activation="tanh")(x)
        x = Dropout(0.2)(x)

        output = Dense(self.class_len, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=output)
        return model
