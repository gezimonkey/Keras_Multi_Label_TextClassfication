MODEL_PATH='saves/model.h5'
TOK_PATH='saves/tokenizer.pickle'
LOG_PATH='./logs/fit/'
EMBEDDING_PATH = '../../dataset/glove/glove.6B.300d.txt'
TRAIN_PATH='data/train.csv'
PREDICT_PATH='data/train.csv'
PREDICT_LEVEL=0.5
# TRAIN_PATH='data/pre_data.csv'
# PREDICT_PATH='data/pre_data.csv'



# CLASSES = ['政治、法律', '社会科学总论', '文化、教育、体育', '语言、文字', '医药、卫生', '计算机科学',
#            '环境科学、安全科学', '历史、地理', '数理科学和化学', '工业技术', '综合、工具书', '文学', '经济、商业',
#            '航空、航天', '建筑科学', '生物科学', '哲学、宗教', '天文学、地球科学', '生活服务技术', '交通运输', '农业科学', '艺术', '军事、战争', '自然科学总论']
CLASSES=['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance']

MAX_SEQUENCE_LEN=256
MAX_WORDS_LEN=20000
EMBED_SIZE=300

CNN='CNN_MODEL'
CHAR_CNN='CHAR_CNN'

FAST='FAST'

TEXT_LSTM='TEXT_LSTM'
TEXT_BI_LSTM='TEXT_BI_LSTM'
TEXT_ATT_BI_LSTM='TEXT_ATT_BI_LSTM'

TEXT_GRU='TEXT_GRU'
TEXT_BI_GRU='TEXT_BI_GRU'
TEXT_ATT_BI_GRU='TEXT_ATT_BI_GRU'

