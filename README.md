选择参加 Quora 这场比赛主要是为了学习一下 NLP（自然语言处理）。受益于广大 kagglers 无私的分享，收获挺大。

先简单总结一下我的方案，主要使用了两种 model：
 1. 梯度提升（LightGBM）
 2. 深度学习（LSTM with concatenation 和 Decomposable attention）

值得一提的是深度学习在这个比赛中，效果还是很惊艳的，避免了特征工程，端到端的独特优势。不过在 Quora 比赛中受限于 pretrained embedding 词汇量覆盖范围，数据集中也存在不少拼写错误，端到端的深度学习 很快就到达了瓶颈。不过在最后用 deep model 和 lgb model 做 stacking、用 deep model 给 lgb model 做特征提取，或者用 deep model concate 一些 特征，都取得了不错的结果。

# 简介
下面简单介绍一下背景。这次比赛的主办方 Quora 是美国一个著名的知识问答网站。Quora 一个重要的产品原则是，尽量每个页面都是不同的问题，避免问题的回答者需要回答重复的问题；也便于搜索答案的用户在一个页面就能获取需要的信息。举一个简单的例子：“What is the most populous state in the USA?” 和 “Which state in the United States has the most people?” 这两个问题虽然表述不同，但有相同的语义，应该在一个问题页面出现。

# Data Exploration
## 数据集
Quora 训练数据集有40多万个可能相似的问题对。如下：  
![](https://leanote.com/api/file/getImage?fileId=5950739bab64416f6a000c65)  
由问题id（qid1，qid2），问题（question1，question2），和是否重复（is_isduplicate）三部分组成。

## 评价指标 evaluation
输出结果为是否是重复问题的概率，使用 log loss 作为评价标准，数学表达式如下：  
![](https://leanote.com/api/file/getImage?fileId=5950739bab64416f6a000c66)  
经常被用作二分类问题的评价指标还有：

 - Precision：P=TP/(TP+FP)
 - Recall：R=TP/(TP+FN)
 - F1-score：2/(1/P+1/R)
 - ROC/AUC：TPR=TP/(TP+FN), FPR=FP/(FP+TN)

## 不平衡的 train，test 数据集
我们来看一下 Quora 比赛中的第一个小“坑”：train 和 test 中正负例所占的比例是不同的。

首先，我们可以计算出 train 数据集正例的平均值为 p = 0.3692。以常量 0.3692 为预测结果做一次提交，得到在 test 上的 Public LB 分数为 0.55。计算 train 上的 log loss，结果为 0.6585。可以看出 train，test 数据集分布是不均衡的。

下面我们计算一下 test 数据集正例的均值。根据：
 - 二分类 log loss 的定义
 - p = 0.3692 时，logloss 为 0.554

 可得：  
![](https://leanote.com/api/file/getImage?fileId=5952131eab6441560e0013a3)  
r 为正例比例，r = 0.174。以常量 0.174 再次提交，Public LP 为 0.463。
可以看出，test 数据集中重复问题的比例比 train 中小了不少。那么如何解决这个不平衡的问题呢？Quora 比赛中大家主要使用了三种方法：
 - 对 train 中非重复问题对（负例）进行过采样，使 train 中正例比例变为 0.174
 - 通过一个转换函数，将 train 上训练模型的预测结果转换到 test 数据集的分布上
 - 通过 Keras 的 class_weight 设置权重
 
方法1，过采样会增大训练集，分割训练集和验证集的时候需要注意对分割后的数据集分别过采样，避免过采样的数据泄露。我在本次比赛中主要使用了方法2，如下：
 - a = 0.174 / 0.3692, b = (1 - 0.174) / (1 - 0.3692)
 - f(x) = a * x / (a * x + b * (1 - x))
通过转换，f(0) = 0; f(1) = 1; f(0.3692) = 0.174  
![](https://leanote.com/api/file/getImage?fileId=59522cbdab644153e30015fe)  
关于推导和更详细的分析可以参考：
[How many 1's are in the Public LB?](https://www.kaggle.com/davidthaler/how-many-1-s-are-in-the-public-lb)
[Statistically valid way to convert training predictions to test predictions](https://www.kaggle.com/c/quora-question-pairs/discussion/31179)

# 特征工程
特征工程是机器学习的基础，同时也非常繁重的一项工作。需要有一定的业务知识和对数据的洞察力。特征工程有多重要呢？引用一句话“数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限”；再引用一句话“Garbage in, garbage out”。在大部分 Kaggle 比赛中算法是比较固定的，而特征工程是决定成绩最关键的因素。

什么是特征工程？我们看一下下面这张图的总结，参考[链接](https://www.zhihu.com/question/29316149)。  
![](https://leanote.com/api/file/getImage?fileId=59547652ab644175ca0009f3)  
Quora 比赛中我使用的特征有200个左右，主要可以分为两部分：

 - nlp 特征，问题文本挖掘特征，比如：问题对字符长度差，问题对相同单词个数（有/无Tfidf权重），问题对 Embedding 的各种 distance，等等
 - 网络特征，由 train、test 问题对组成的 graph 提取的各种特征（问题为 node，问题对为 edge），比如：问题出现频率，问题对中两个问题共同邻居的个数，问题的 PageRank，等等。网络特征在 Kaggle 论坛引起了不少争议，大部分都认为这是 Quora 以某种规则构造本次比赛数据集导致的 leakage。

## nlp 特征
文本长度特征：
 - 字符长度、长度差、比例
 - 单词长度、长度差、比例
 - 平均单词长度、差值
 - 首字符大写单词个数、差值

相同单词特征：
 - 相同单词个数、比例
 - 增加 Tfidf 权重相同单词个数、比例
 - 去除停用词相同单词个数、比例
 - 相同名词个数、比例
 - 去除停用词相同 unigrams、bigrams、trigrams 单词个数、比例
 - 词干化相同单词个数、比例

Embedding 特征（分别使用了 glove，google news，fasttext）：
 - cosine/euclidean/jaccard/minkowski/... distance
 - word mover's distance，关于 wmd 简单的介绍可以参考[链接](https://www.zhihu.com/question/29978268?sort=created)

Fuzzy 特征（基于 [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy)）：
 - QRatio、WRatio、partial ratio...

SimHash 特征（基于 [simhash](https://github.com/leonsim/simhash)）:
 - word/word bigrams/word trigrams distance
 - character bigrams/character trigrams distance

Tfidf Vector/Counter Vector 特征：
 - tfidf mean/sum/len
 - 基于 Counter Vector（char）的 unigram/bigram/trigram jaccard 距离
 - 基于 Tfidf（char）的 unigram/bigram/trigram euclidean 距离
 - 基于 Tfidf（char）的 oof（out of fold）特征
 - 基于 Tfidf（char）的 SVD（[Singular value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)）、SVD oof、等特征
 - 基于 Tfidf（word）的 euclidean 距离
 - 基于 Tfidf（word）的 oof 特征

POS（part of speech）、NER（named entity recognizer）特征，基于 Stanford CoreNLP:
 - Noun（n），Adjective（a），Verb（v），Personal Pronoun（prp），WH-Pronoun（wp），Numbers（cd） 等 POS 特征（参考[标签](https://stackoverflow.com/questions/1833252/java-stanford-nlp-part-of-speech-labels)）
 - NER 匹配特征

上面不少特征都来自于热心 kagglers 的分享，参加 Kaggle 比赛一定要经常关注论坛，比赛结束后 top teams 会分享一些他们的取胜方法，甚至直接在 github 开源。

## 网络特征（magic feature）
网络特征可能是这次 Quora 比赛争议最大的部分了。网络特征效果之好，以至于大家都觉得参加了一个假 NLP 比赛。

记得当时比赛进行了一多半的时候，我的成绩在0.20+（下面提到分数都是 Public LB），和 top teams 的成绩差距非常大（0.13+）。想方设法增加 nlp 特征，改进 model，却只能得到很有限的提升。后来，两个 kagglers 分享了他们发现的 magic feature：
 - 问题出现在数据集中的频率越高，是重复问题的概率越大；
 - 问题对中的两个问题，共同邻居越多（已知问题对 a - b，则 a 和 b 互为邻居），是重复问题的概率越大。

下面我们来挖掘一下这两个特征背后隐含的意义。首先来看一下这次比赛的数据构成：问题对，train，test 数据集每一行都是一个问题对。
我们知道的是自然状态下问题不是成对出现的，必然是通过某种方式组合起来。但是这些问题对是如何构成的呢，我们不得而知，Quora 在最后也没有回应这个问题。不过在 Quora 官方的[博客](https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs)中，可以发现一些蛛丝马迹：

> Therefore, we supplemented the dataset with negative examples. One
> source of negative examples were pairs of “**related questions**” which,
> although pertaining to similar topics, are not truly semantically
> equivalent.

可以看出问题对的构成和“相关问题”有关，比如用同一个标签（e.g.社会学）下的几个问题两两比较来生成问题对。而上面两个特征应该体现了 Quora 生成问题对背后某些规律。

如何挖掘这类特征呢？可以把数据集中出现的每个问题看做一个节点（node），每个问题对看做节点的边（edge）。如此，就构成了网状结构，如下展示的是其中一个子图（[参考](https://www.kaggle.com/davidthaler/duplicates-of-duplicates)）：  
![](https://leanote.com/api/file/getImage?fileId=59562829ab64414a81001826)  
虽然在实际 NLP 问题中这些网络特征可能没什么用，但是从 Quora 这个问题网络中，还是可以发现很多有意思的东西。比如，可以求出每个问题的 PageRank，可以找出每个子图中的节点、边数等等。而两个节点所在的团（[clipue](https://en.wikipedia.org/wiki/Clique_%28graph_theory%29)）大小最后被证明为最有效的 magic feature。

# Models
在比赛中主要使用了两种 model：梯度提升和深度学习。梯度提升用的 LightGBM，比 xgboost 速度快了不少；深度学习用的 Keras，backend 是 TensorFlow。
使用了全局的 5 Fold 做 validation，便于 Stacking。
## LightGBM
梯度提升是一种基于决策树的算法，在 Kaggle 比赛中可以算是明星算法，出镜率非常高，而且经常是作为 best single model。这得益于其准确率高，速度快，而且对特征中的各种空值、inf值不挑剔，使用起来非常方便。[LightGBM](https://github.com/Microsoft/LightGBM) 是微软开源的一个高性能的梯度提升实现。也是我在 Quora 比赛中的主力，分别用来训练 L1 base model，和 L2 stacking。

### 参数调优
 - num_leaves：树的子叶数，用来控制树的复杂度。
 - min_data_in_leaf：子叶节点中最少数据个数，提高此值可以降低 over-fitting 风险
 - learning_rate：学习速率
 - feature_fraction：每次迭代随机选择一部分 feature 进行训练，可以提高训练速度，降低 over-fitting 风险
 - bagging_freq：每迭代 k 次，做一次 bagging，和 bagging_fraction 一起生效
 - bagging_fraction： 每次迭代会随机选择部分 data，进行训练，提高训练速度，降低 over-fitting

具体可见 [Parameters Tuning](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters-tuning.md)，[Parameters](https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.md)
## Deep Models
Deep Learning 这几年在 NLP 领域快速发展，机器翻译、看图说话、自动摘要各种技术层出不穷。有不少 kaggler 都是想知道 Deep NLP 判断语义相似度的表现来参赛的，Deep NLP 在 Quora 比赛中也没有让大家失望。端到端的学习方式，避免了费时耗力的特征工程。下面介绍一下我在比赛中使用的两种 deep model。另外，比赛结束后看到有 kaggler 使用 1D CNN 获得了非常不错的成绩，这里也一起介绍下。

### LSTM with concatenation
先来看一下 model 的结构，如下图（来自 Quora [博客](https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning)）  
![](https://leanote.com/api/file/getImage?fileId=5957047dab64412309000c2c)  
分别使用了 [GloVe](https://nlp.stanford.edu/projects/glove/)、[Google News](https://code.google.com/archive/p/word2vec/)、[fasttext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) 训练好的词向量（也尝试了用 train，test 数据集中的问题作为语料来训练词向量，效果一般）。用这些词向量来生成问题对的 embedding。将问题的 embedding 输入 LSTM 层，得到两个问题的 representation，简单拼接后，输入 Dense layer，得到分类结果。
简单的 5 fold（reweighted）可以得到 0.26 左右的成绩。稍微改一下结构，拼接一些前面提到的 NLP 特征、网络特征，可以达到 0.14 左右。需要注意特征值的标准化和缺省值处理。

### Decomposable attention
这个 model 基于 attention 机制（来自 Google Research 论文[链接](https://arxiv.org/abs/1606.01933)），结合了 attention 和 token alignment。decomposable attention 最大的优势是，参数少，和上面 model 相比训练效果没有降低，训练速度提高了一倍。它通过问题对中的每一对单词来训练一个 attention model（soft alignment），然后比较对齐后的短语，最后汇总比较结果来判断是否问题对重复，如下图（来自[链接](https://arxiv.org/abs/1606.01933)）：  
![](https://leanote.com/api/file/getImage?fileId=5958b5f2ab644174f30012ee)  
同样可以通过 concate 特征提高成绩。

### 1D CNN
除了 LSTM，CNN 也经常被用在 Deep NLP model 中。LSTM 由于需要串行计算输入的 embedding，性能比 CNN 低一些。而 CNN 通过使用不同的 kernel size 来建模不同距离的关系（kernel size 为 3，可以看做 trigrams words），但对长距离的依赖比较难处理。Quora 数据集中问题中单词的个数均值约为 11，std 约为 6，所以 CNN 在 Quora 数据集上的表现还是很好的。

Quora 比赛具体实现可以参考[链接](https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23)。问题对 embedding 后，分别输入不同 kernel size（1 ~ 6） 的卷积层，每个卷积层输出经过 GlobalAveragePooling1D 池化层压缩，然后计算两个问题的 absolute difference 和 multiplying（没有直接 concatenate），最后通过 Dense 层，得到问题对匹配结果。

# Ensemble
[Ensembel learing](https://en.wikipedia.org/wiki/Ensemble_learning) 是通过训练多种不同的机器学习算法，分别作出预测，然后通过某种方式（比如加权平均、stacking、bagging 等等）将这些预测结果结合起来作为最终结果。单个 model 的预测结果越准确，同时不同 model 间的相关性越低，ensemble 效果越好。

Ensemble 是 Kaggle 比赛中非常重要的一个环节。top team emsemble 几十上百 model 非常常见。我在 Quora 比赛中训练了 10 个左右 model，使用了 stacking 和加权平均两种 emsemble 方法。先介绍一下 stacking，简单的说就是通过训练一个机器学习模型来总结几个机器学习模型的预测结果，然后做出最终预测。下图（来自[链接](https://dnc1994.com/2016/04/rank-10-percent-in-first-kaggle-competition/)）展示了一个 5 fold stacking 例子：  
![](https://leanote.com/api/file/getImage?fileId=5959c835ab644133a3000baf)  
在 4 个 fold（蓝色矩形）上训练第一层 model，然后在 1 个 fold（橙色矩形）上预测结果。进行 5 次训练后，将得到的 5 个预测结果（5 个橙色矩形）组合起来，作为上层 model 的一个 feature。关于 emsemble 详细介绍可见[链接](https://mlwave.com/kaggle-ensembling-guide/)。

比赛中我的第一层 model 使用了 1 个 lgb model，2 个 lstm concate features model，2 个 decomposable attention concate features model，第二层使用了一个 lgb model 来 stacking，得到结果 A（0.127+，Public LB）。另外，还训练了 1 个 lgb model，加入了 lstm 和 decomposable attention oof prediction 作为特征，得到结果 B（0.128+）。A 和 B pearson 相关性约为 0.98，取平均值得到最终结果（0.125+）。也试了对 A 和 B 做一次第三层 stacking，结果并不理想。用 deep model oof prediction 作为特征，是结束前两天才尝试的，没想到能得到和 stacking 差不多的结果。

# Postprocess
在提交结果前，对预测结果做了前面提到的转换 f(x) = a * x / (a * x + b * (1 - x))，来平衡 train，test 分布的差异。

最后说一下比赛结束后论坛里分享的一个有意思的 postprocessing（[链接](https://www.kaggle.com/divrikwicky/semi-magic-postprocess-0-005-lb-gain)）。作者基于，如果问题 A 和 B 重复，问题 B 和 C 重复，则问题 A 和 C 也应该重复的传递性，对预测结果做了重新校准。用我最好的提交简单套用了一下，重新提交提高了0.04+（0.121+ Public LB，0.125+ Private LB）。看了一些 top team 分享，有的是直接对这个传递性进行特征提取，然后 stacking，也是效果拔群。最开始的时候利用过问题的传递性来扩大训练集（a = b，b = c 则 a = c；a ！= b，b = c，则 a ！= c），没有提高就放弃了，最后竟然被用在了这里。

# Lessons Learned
看了一些 top team 分享的方案，感觉自己还可以提高的地方：
 - 训练更多种类的 model，比如 random forest、knn 等，增加多样性
 - deep model 输出不是特别稳定，可以尝试在每个 fold 里用不同的 seed 多 bagging 几次
 - 拼写检查（比赛中使用了 PyEnchant，没有感觉到有明显提高，放弃了）、文本清理
 - 尝试基于字符的 deep model，对于拼写错误应该有更高的抵抗力，增加 model 多样性

Quora 比赛分享方案汇总，有兴趣的同学可以看下[链接](https://www.kaggle.com/c/quora-question-pairs/discussion/34325)
# 参考

 - [LSTM with word2vec embeddings](https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings)
 - [How many 1's are in the Public LB?](https://www.kaggle.com/davidthaler/how-many-1-s-are-in-the-public-lb)
 - [Magic Features (0.03 gain)](https://www.kaggle.com/jturkewitz/magic-features-0-03-gain)
 - [Duplicates of Duplicates](https://www.kaggle.com/davidthaler/duplicates-of-duplicates)
 - [PageRank on Quora - A basic implementation](https://www.kaggle.com/shubh24/pagerank-on-quora-a-basic-implementation)
 - [1D CNN (single model score: 0.14, 0.16 or 0.23)](https://www.kaggle.com/rethfro/1d-cnn-single-model-score-0-14-0-16-or-0-23)
 - [Semi-Magic PostProcess (0.005 LB gain)](https://www.kaggle.com/divrikwicky/semi-magic-postprocess-0-005-lb-gain)
 - [Another data leak](https://www.kaggle.com/c/quora-question-pairs/discussion/33287)
 - [0.29936 solution](https://www.kaggle.com/c/quora-question-pairs/discussion/32313)
 - [Is That a Duplicate Quora Question?](https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur)
 - [Keras Decomposable Attention](http://git.iguokr.com/mobile_group/Onigiri_IOS/merge_requests/291)