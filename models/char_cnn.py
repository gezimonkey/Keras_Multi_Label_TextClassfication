from tensorflow.keras.layers import Input, Convolution1D, GlobalMaxPooling1D, concatenate, Dense, AlphaDropout, Concatenate
from tensorflow.keras.models import Model


class CHAR_CNN:
    def __init__(self, embedding_layer, class_len, max_sequence_len):
        self.embedding_layer = embedding_layer
        self.class_len = class_len
        self.max_sequence_len = max_sequence_len

    def build(self):
        conv_layers = [[256, 10], [256, 7], [256, 5], [256, 3]]
        fully_connected_layers = [1024, 1024]
        input = Input(shape=(self.max_sequence_len,),
                      dtype='int32', name='input')
        embedded_sequence = self.embedding_layer(input)

        convolution_output = []
        for num_filters, filter_width in conv_layers:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(embedded_sequence)
            pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(
                num_filters, filter_width))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        for fl in fully_connected_layers:
            x = Dense(fl, activation='selu',
                      kernel_initializer='lecun_normal')(x)
            x = AlphaDropout(0.5)(x)

        output = Dense(self.class_len, activation='sigmoid')(x)
        model = Model(inputs=input, outputs=output)
        return model
