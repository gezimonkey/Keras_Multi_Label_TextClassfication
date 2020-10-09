from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, Dropout, Flatten, Dense, concatenate
from tensorflow.keras.models import Model


class CNN_MODEL:
    def __init__(self, embedding_layer, class_len, max_sequence_len):
        self.embedding_layer = embedding_layer
        self.class_len = class_len
        self.max_sequence_len = max_sequence_len

    def build(self):
        filter_sizes = [1, 2, 3, 4, 5]
        input = Input(shape=(self.max_sequence_len,),
                      dtype='int32', name='input')
        embedded_sequence = self.embedding_layer(input)

        conv_layers = []
        for fsz in filter_sizes:
            conv1 = Conv1D(
                256,
                fsz,
                kernel_initializer='lecun_uniform',
                activation='tanh',
            )(embedded_sequence)
            pool_size = self.max_sequence_len - fsz + 1
            pooling = MaxPooling1D(pool_size=pool_size)(conv1)
            conv_layers.append(pooling)
        merged = Concatenate()(conv_layers)
        dropout = Dropout(0.5)(merged)
        flattened = Flatten()(dropout)

        output = Dense(self.class_len, activation='sigmoid')(flattened)
        model = Model(inputs=input, outputs=output)
        return model

    def build_base(self):
        filter_sizes = [3, 4]
        convs = []
        input = Input(shape=(self.max_sequence_len,),
                      dtype='int32', name='input')
        embedded_sequence = self.embedding_layer(input)

        for fsz in filter_sizes:
            conv = Conv1D(512, kernel_size=fsz, activation='relu')(
                embedded_sequence)
            pool = MaxPooling1D(2)(conv)
            convs.append(pool)
        merge1 = concatenate(convs, axis=1)
        dropout = Dropout(0.5)(merge1)
        conv1 = Conv1D(256, 4, activation='relu')(dropout)
        pool1 = MaxPooling1D(5)(conv1)
        flat = Flatten()(pool1)
        dense = Dense(512, activation='relu')(flat)

        output = Dense(self.class_len, activation='sigmoid', name='pred')(dense)
        model = Model(inputs=input, outputs=output)
        return model
