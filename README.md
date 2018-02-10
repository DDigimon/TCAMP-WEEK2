# TCAMP-WEEK2
好未来第二周
DL-naive
第二周内容
自动评分
第一周是句子相似度计算，代码地址：
https://github.com/qiangzi11hao/Semantic-Textual-Similarity

# 1. Data
这次数据集提供围绕特定主题的英语文章文本数据，可用于针对学生所写的英文作文进行机器自动评分。
由于对于应试教育作文打分有一定规则，因而可以通过机器来学习这样的规则。每个老师对于作文评分
在1至6之间，一篇作文由两位老师打分，之后相加得到作文总分。最后提交为这篇作文总分，作为预测结果。

项目提供txt文件，每列由tab隔开。每行为一个样例
训练集共1200个样本，5个字段，分别为编号，文本，评分1，评分2，作文总分

示例：

1 "Dear Local Newspaper, I believe the ……” 4 4 8

2 "Dear @CAPS1 @CAPS2, I have heard ……” 6 4 10

测试集共550样本，2个字段，分别为编号，文本。

示例：

1 "Dear Local Newspaper, I believe the ……”

2 "Dear @CAPS1 @CAPS2, I have heard ……”

## 实验结果
![image text](https://github.com/DDigimon/TCAMP-WEEK2/blob/master/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/233.png)

# 2 特征提取
## 2.1 Length Features
首先我们先统计字符数、句子数以及平均句子长度。以下不做特殊说明，大家可以把每个函数中的content当作是我们training set中的单个essay!
## 2.2 Occurrence Features
统计每个essay中的标点、stopwords、错词以及SAT词汇出现的次数。这里SAT词汇来自是将雅思官网5000词汇down下来后，存到了本地了txt文件中，然后通过定义的get_sat_word 函数进行提取。
## 2.3 word ngrams Features
首先，我们统计了word ngrams（n=1,2,3）,通过使用python中的set函数，我们可以去将grams中重复的部分去掉，
这样就可以直观地表示出每个essay词汇量大小（还包含文章间的词汇的组合差异情况）。如果一个文章中出现的重复词太多，那必然会导致在该特征下的表现值较小。（虽然开始的时候是想做ngrams统计的，但是...）
## 2.4 Pos Features 
这里我们统计名词、动词、介词等各种词性的数量，一篇好的essay在各种词性上必然有着一定的统计规律。
## 2.5 Sentiment Features
情感分析一直以来都是NLP的一个重大课题，这里通过调用NLTK的sentiment相关函数，对essay进行分析。对于不同的文章，所需的情感往往是不同的，如叙述性就需要我们的情感饱满些，议论文则需要我们客观些。
## 2.6 Readability Features
参阅Task—Independent Features for Automated Essay Grading，我们建立了一系列基于单词、字母、音节以及其他各项指标的可读性（readability）特征（包括了Flesch,Coleman_Liau,ARI,SMOG以及Fog等等）。这些都是特征能够很好评定一个文章的可读性，而一篇高分的essay必然有着属于自己的各项readability指标组合。通过引入这些特征我们希望可以拟合出一个适用于评分的标准组合。

以上的所有特征，我们称它们为Dense Featurs，特征工程已经完成了80%，与之对应的Sparse Features自然就是那剩下的20%啦。或许你会疑惑：为啥是20%，有什么统计依据吗？问得好！并没有...都是个人感觉。
由于后面引入Sparse Features 会导致我们维数发生爆炸，自然而然训练时间也会随之增加。
在这里我们做了一张各个特征的之间相关性的统计图，对各个特征进行了梳理，方便在后面的模型中进行选择
![image text](https://github.com/DDigimon/TCAMP-WEEK2/blob/master/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/114163400.jpg)
到这里我们已经完成了所有的dense特征的提取工作，下一步就是提取它们的Sparse Features。

## 2.7 Sparse Features [ POS ngrams]
我们考虑了POS 的unigrams，bigrams，和trigrams。essay通过Stanford part-of-speech tagger进行标注，然后调用了Sklearn中的CountVectorizer进行向量化。特别指出，对于那些在所有essays出现不超过10次的feature，都会被直接过滤。最后我们得到了一个3888维的向量！

# 3. 模型选取及相关结果

模型方面我们选用了Random Forest、XGboost、GradientBoosting和Lasso CV 四个模型，最后将四者的结果求平均作为我们结果的输出值。但在某个交叉集中，我们总是可以找到某一个模型优于其他的，但是在最后的test中，我们总是以平均后的结果得到最高分。我们得出：集成的模型鲁棒性更好。
我们采用了前向叠加网络的方式，对我们的特征进行了分析，每次只增加一个特征，然后观察其在验证集上的表现。
![image text](https://github.com/DDigimon/TCAMP-WEEK2/blob/master/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/1416796153.jpg)
可以发现，有些特征的表现十分突出，如第七个特征，而在之前的特征相关性图中，第七个特征它与其他的特征的相关性很低，说明往往相关性很低的特征可以为系统的预测带来不错的贡献。但是对于一些会使pearsonr下降的特征我们仍保持怀疑态度，因为在最后的提交过程中，我们发现，所有特征都用上的时候，得分是最高的。所以说，在验证集上出现的下降现象并没有发生，反而是特征不足的情况下效果不佳。由于Sparse 特征维数过大，我们这里没有对其进行画图统计，但是最后的结果依然适用：加稀疏特征比不加好！

# 4.深度学习 方法

## 4.1模型简介
神经网络部分主要分为对文本本身的词向量进行训练和经过特征提取的简单网络结构组成。词向量部分使用glove.840B.300d作为预训练词向量信息，对于词向量库里没有的，重新基于文档进行训练。特征部分来自队友整理的18组特征，归一化以后进入网络训练。

对文本进行训练时选择3层cnn卷积网络来抓取文本特征，之后使用一个LSTM模型衔接，达到文本特有的文本序列的训练效果。实验中发现仅使用多层cnn pearson能够达到百分之70以上，加入LSTM能够达到百分之80以上。

单独使用特征进行网络学习效果不佳，最高pearson系数仅有63%，与文本网络融合以后影响也不是特别显著。最后深度学习网络最佳成绩仅有84%

语言晦涩，直接上图
![image text](https://github.com/DDigimon/TCAMP-WEEK2/blob/master/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/111.png)

## 4.2实验结果
如图所示是神经网络训练loss值与其训练阶段的关系图。可以发现随着训练的时间加长，训练集的loss值在下降，而验证集的loss值已经出现趋于平稳状态。这表明模型是一个未完全拟合的模型，但是却感觉没有办法深入调整网络结构。
![image text](https://github.com/DDigimon/TCAMP-WEEK2/blob/master/%E4%BC%A0%E7%BB%9F%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%96%B9%E6%B3%95/2222.png)
## 4.3 结果分析
深度学习方法由于训练量过小，并不能完全体现优势，虽然能够提取出文本特征，但发现文本特征并不适合与深度学习融合。学习结果中出现不应该出现的。如作文最低分有2分，但机器判断却只有0.38分。这是不符合范围的数据。也就是说，模型还不能较优的拟合所有状态。如果问题是分类问题，兴许会好些，但是没有时间实验了。

