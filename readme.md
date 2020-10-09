# Keras下的多标签分类(Keras Multi Label TextClasscation)

## 简介(Introduction)

利用KERAS结合不同的分类器，对英文文本进行多标签分类，对英文图书简介进行中图法分类，最终将简介分为24个一级分类

## 依赖(requirements)  

tqdm==4.49.0  
numpy==1.18.5  
pandas==1.1.2  
tensorflow==2.3.0  
matplotlib==3.3.2  
pyenchant==3.1.1  
scikit_learn==0.23.2  

## 数据集(Dataset)  

数据集来自[Kaggle](https://www.kaggle.com/vetrirah/janatahack-independence-day-2020-ml-hackathon)  
Dataset from [Kaggle](https://www.kaggle.com/vetrirah/janatahack-independence-day-2020-ml-hackathon)  

清洗后的数据概况(Dataset detail after cleaning)  
MAX SEQ LEN:778  
ALL WORDS:20430  
DATA LEN:20971  
| 分类(Class)          | 数量(Quantity) |
| :------------------- | :------------- |
| Computer Science     | 8594           |
| Physics              | 6013           |
| Mathematics          | 5618           |
| Statistics           | 5206           |
| Quantitative Biology | 586            |
| Quantitative Finance | 249            |

## 所使用的分类器及成绩(Result)

| 分类器(Classifier) | val_categorical_accuracy | epochs |
| :----------------- | :----------------------- | :----- |
| CNN                | 0.7607                   | 18     |
| FAST               | 0.7417                   | 13     |
| **CHAR_CNN**       | **0.7655**               | **10** |
| TEXT_ATT_BI_GRU    | 0.7493                   | 5      |
| TEXT_ATT_BI_LSTM   | 0.7583                   | 15     |
| TEXT_BI_GRU        | 0.7564                   | 5      |
| **TEXT_BI_LSTM**   | **0.7583**               | **14** |
| TEXT_GRU           | 0.7626                   | 19     |
| TEXT_LSTM          | 0.7498                   | 4      |

虽然CHAR_CNN的正确率看起来高那么一点点，但实际上TEXT_BI_LSTM的结果更接近真实数据  

### 分类结果(TEXT_BI_LSTM)

| 分类(Class)          | 数量(Quantity) |
| :------------------- | :------------- |
| Computer Science     | 8526           |
| Physics              | 5603           |
| Mathematics          | 5464           |
| Statistics           | 5640           |
| Quantitative Biology | 305            |
| Quantitative Finance | 229            |

## 参考(Reference)

[Magpie](https://github.com/inspirehep/magpie)

[Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification)
