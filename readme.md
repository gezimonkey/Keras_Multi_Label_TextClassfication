# Keras下的多标签分类(Keras Multi Label TextClasscation)

## 简介(Introduction)

* 本项目利用KERAS结合不同的分类器并使用Glove词向量，对英文文本（中文需修改load_data.py增加分词功能并更换词向量）进行多标签分类。  
* 项目起源是因为以前做过美亚图书分类转换中图法分类，当年是通过爬取美亚的图书信息和分类，然后人工制定规则映射，时间久了积攒了些数据，就想着拿KERAS实现一下AI智能分类，毕竟还有很多图书信息不是来自美亚，没有对应的分类。  
* 共有9种分类器可供选择，大部分模型只做到了baseline,部分借鉴了[Magpie](https://github.com/inspirehep/magpie)和[Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification)。  
* 数据质量及标签数量对分类结果有非常大的影响，数据太过稀疏及标签过多会导致结果惨不忍睹（3分类ACC可达92%，6分类ACC76%，24分类则只有41%）。

  The English introduction translated from google，good luck...
* This project uses KERAS and Glove to combine different classifiers to classify English text (Chinese need to modify load_data.py to add word segmentation and change the Embedding) for multi-label classification.
* The origin of the project is that I have done the Amazon Book classification to CLC (Chinese Library Classification) many years ago. In the past, it was by crawling the book information and classification from  Amazon.com, and then manually formulating the rule mapping. After a long time, I accumulated some data, and I wanted to use KERAS to implement AI. Intelligent classification, after all, there is still a lot of book information that does not come from Amazon.com, and there is no corresponding classification.
* There are a total of 9 classifiers to choose from, most of the models only achieve the baseline, and partly draw on [Magpie](https://github.com/inspirehep/magpie) and [Keras-TextClassification](https://github.com/yongzhuo/Keras-TextClassification)。
* The quality of data and the number of tags have a very large impact on the classification results. Too sparse data and too many tags will lead to disastrous results (ACC 92% for 3 categories, 76% for 6 categories, and 41% for 24 categories).

## 依赖(Requirements)  

tqdm==4.49.0  
numpy==1.18.5  
pandas==1.1.2  
tensorflow==2.3.0  
matplotlib==3.3.2  
pyenchant==3.1.1  
scikit_learn==0.23.2  

## 使用方法(Guide)

1.配置config.py中的参数(config config.py)  
2.修改train.py的模型类型后运行 (modify train.py and run)  
3.修改predict.py的模型类型后运行(modify train.py and run)

也可以直接运行 run.py(Can also run run.py directly)

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

## 分类器及成绩(Result)

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

虽然CHAR_CNN的正确率看起来高那么一点点，但实际上TEXT_BI_LSTM的结果更接近真实数据。  
Although the correct rate of CHAR_CNN looks a little bit higher, in fact the result of TEXT_BI_LSTM is closer to the real data

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

## TODO

1.针对多标签的稀疏数据做SMOTE平滑处理。 (SMOTE for multi label sparse data)  
2.找到更大，更多标签的数据集。 (Find larger, more tagged datasets.)  
3.模型优化(Model optimization)  
