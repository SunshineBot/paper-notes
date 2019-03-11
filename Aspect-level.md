最近看了几篇文章，记一下大概内容

1. ACL2014_Dong et al.2014_Adaptive recursive neural network for target-dependent Twitter Sentiment Classification
算是比较早的推特target-dependent情感分类的文章了。这篇文章的主要内容就是提出了一个基于递归神经网络的特征构建方法。
先进行句法依存分析，然后从句子中的target词的embedding开始，按照句法依存关系一步一步递归的整合相邻的两个词的embedding，最终将整个句子embedding出来，再用分类器进行分类。
倒是让我了解了递归神经网络和循环神经网络。为了解决不定长的问题，循环神经网络把输入视为一个序列，依次读入数据并更新hidden state，递归神经网络把输入视为一个树结构，自顶向下/自底向上依次读入数据并递归到树的下/上一层。

2. IJCAI2015_Vo and Zhang2015_Target-dependent twitter sentiment classification with rich automatic features
这篇文章最大的特点就是结合了target的上下文。其实结合的方式非常简单，就是把target词的左边作为上文，右边作为下文，分别进行embedding作为特征。还提出了一个整合了情感词典作为先验信息的embedding方法，其实就是删掉情感词典中没有的词对应的embedding并填充0作为一个新特征。
文章中使用了word2vec和SSWE两种embedding方法，还尝试整合这两种方法。发现SSWE效果比word2vec好1%，整合效果比SSWE好约1.5%。
文章中还探讨了几种不同的池化方法的效果。

3. ICLR2017_Learning End-to-End Goal-Oriented Dialog
这篇论文用memory network做了一个goal-oriented对话系统，所谓goal-oriented，在这篇论文中就是指和顾客对话收集订餐需要的信息然后帮顾客订餐厅这么件事儿。这篇文章提出的目的是想验证端到端的神经网络能不能在对话系统中达到传统规则或者监督方法相同的效果，就真实对话数据而非人工构造数据集的结果来看，效果是明显好于传统方法的，虽然仍然很差。
相比单纯的end-to-end memory network，这篇文章实现的时候有几个不同点：
- memN2N有matrix ABCW总共4个参数矩阵。在这里删除了output matrix C，此外记忆内容和输入是同样来源，所以matrix B和matrix A是同一个矩阵。
- 历史对话的每一个对话直接记为memory中的2条记忆（问&答）
- 多跳网络更新有2种方法，一个是相邻的A和C相等，一个是每跳的输入uk之间加一个变换参数矩阵H，这篇文章使用了第二种方法，并进行了改动，加上了记忆和权重的期望值。
- 传统的记忆网络只能输出一个单词，准确的说是一个向量。在这个对话系统中就做的比较triky，他把所有可能的回答（4000种）都加载进来进行了mebedding，最后从中挑选一句作为回答。

4. COLING2016_Effective LSTMs for Target-Dependent Sentiment Classification
相比于Vo and Zhang2015，这篇文章是将LSTM整合到target-dependent sentiment analysis中来了。论文中对target词汇的上下文的定义，和Vo and Zhang2015中是一样的，将target词左边的词视为上文，右边的词视为下文，以此来区分对不同target进行的情感分析。我还没有看太多的别的论文，但是感觉这篇文正一个很重要的点是提出了一个基于RNN/LSTM的处理target-dependent sentiment analysis任务的端到端的NN模型。今天刚好看了Andrew Ng的LSTM视频，论文中处理上下文信息的方法则和BiLSTM如出一辙，对于某个词，从开头依次从左往右到target词喂入LSTM视为上文，从结尾依次从右往左到target词喂入LSTM视为下文，拼接后进行softmax，就是TD-LSTM模型。而TC-LSTM模型，就是将target词的词向量拼接到每个词后面，作为输入。
结果来看，acc稍高于Vo and Zhang2015，F1-macro稍低于Vo and Zhang2015，总的来看基本是打个平手吧。

5. EMNLP2016_Attention-based LSTM for Aspect-level Sentiment Classification
这篇文章中的模型，除了基本的LSTM，还做了两个改动。一个是题目中提到的Attention-based，每一个时间t的隐层状态h_t构成一个矩阵H(d×N)，再把aspect的embedding vector重复N次，拼接到每个h_t上，构成一个矩阵((d+d_a)×N)，乘以一个参数矩阵[W_v, W_h]做一个tanh变换，然后进行softmax作为权重α，最终的给每个句子生成一个向量r=HαT，在r后面接一个线性变化进行分类。另一个是Aspect Embedding，将aspect的embedding拼接到每个word embedding vector后面作为输入，这个上一篇LSTM的Target Connection是一样的。但是事情似乎并没有这么简单，因为实验发现TC比TD还差，但是这里的AE加上以后效果变好了，文章原文这样评价的：```In our models, we embed aspects into another vector space. The embedding vector of aspects can be learned well in the process of training. ATAELSTM not only addresses the shortcoming of the unconformity between word vectors and aspect embeddings, but also can capture the most important information in response to a given aspect. In addition, ATAE-LSTM can capture the important and different parts of a sentence when given different aspects.```但是Aspect Embedding这块我还没有完全搞懂是怎么实现的，有空看看代码吧
相比前面几篇试图使用拆分上/下文的方法来提供target-context信息，使用attention机制明显更加合理，可以更好的从整个句子的层面对不同的词分配不同的权重，而非仅通过将target词左右内容进行简单拆分，效果也非常好。
这个对所有隐层状态进行attention-based的机制，和memory network中对memory信息进行选取的过程其实是相似的（attention多了一个W_h、W_v的线性变换），不知能否互相借鉴。  
Acc: Restaurant77.2%, Laptop68.9%(not attention-based model)

6. AAAI2017_Attention-Based LSTM for Target-Dependent Sentiment Classification
这篇文章主要提出了两种attention的计算方法：
- 第一种是用了attention network，（好像是）对每个隐层状态h_t与target词的隐层状态h_target做点积再进行softmax（但是参数θ是干啥的？）（有必要看看attention的论文了）
    - Acc: Tweet71.6%
    - F1: Tweet71.2%
- 第二种是用一个双线性项W_b，attention=softmax(h_tT\*W_b\*h_target)，效果比上一个好。受(Chen, D.; Bolton, J.; and Manning, C. D. 2016. A thorough examination of the cnn/daily mail reading comprehension task. In ACL.)启发想出来的点子
    - Acc: Tweet72.6%
    - F1: Tweet72.2%

7. Tang et al., 2016_EMNLP2016_Aspect Level Sentiment Classification with Deep Memory Network
这篇文章另辟蹊径，用memory network来进行aspect level情感分析。不过思路其实是attention机制的思路，给句子中的每个词赋予不同的权重。第一步，将句子中的每个词的word embedding作为记忆中的每一项，计算每个记忆m_i的gi=tanh(W_att \* [m_i, v_aspect] + b_att)，再对gi进行归一化得到权重α_i，最后对记忆加权求和得到vec。第二步，将target词的embedding作为输入部分，进行一次线性变换后与vec相加，作为下一层的输入。最后一层的输出直接接一个softmax层进行分类。每层的参数是共享的，所以其实参数量很小，训练起来非常快（数量级上的变化）。
文章还提到了location attention，意思是位于不同位置的词所占的权重应当是不一样的。这是在将句子中的每个词embed到memory中的时候使用的，对每个m_i进行不同的操作，有逐点相乘也有简单的相加。
原始记忆网络的embedding过程有3个embedding参数矩阵分别对应的input，memory，output，在这里没有output，input和memory直接使用的词向量，因而不需要训练。处理方法和ICLR2017_Learning End-to-End Goal-Oriented Dialog一样。
Acc: Restaurant80.95%, Laptop72.37%

8. EMNLP2017_Recurrent Attention Network on Memory for Aspect Sentiment Analysis
之前还在想memory network怎么和lstm结合在一起，某天晚上睡觉的时候还在想这个，结果这篇论文就这么干了。  
之前的memory network（也包括ICLR2017的那篇Learning End-to-End Goal-Oriented Dialog）在生成memory的时候采用的方法都是直接使用embedding矩阵的方法，为了简化参数，直接将原本的embedding参数矩阵替换为word2vec/GloVe等词向量，利用每个词的词向量（或者基于词向量将句子压缩成一个与词向量同维度的向量）作为每个memory的内容。在EMNLP2016_Aspect Level Sentiment Classification with Deep Memory Network中则使用memory作为attention，而在lstm中则使用隐层状态作为attention。这篇文章则结合二者，将memory形成过程从简单的查询词向量，变成使用lstm来获得隐层状态，再通过position weighted操作形成memory，放入memory network进行训练。  
memory network的一个优点是参数量很少（MemN2N）只有embedding矩阵ABCW（如果使用词向量简化，可以去掉ABC，如果用作分类，可以去掉W），多层情况下仅多了一个变换矩阵H（embedding矩阵权值共享的训练方式），*我感觉*在加入其它复杂组件的时候，并不容易导致网络难以训练。
这篇文章的position weighted还多加了1维，用来表示target词在文章中的相对位置。  
虽然看起来十分像memory network，但是作者从RNN的角度来看待这个模型。其实它相当于一个多时间片的循环网络（GRU），输入是e_0，输出依次是e_1、e_2、...、e_n，每个cell还有一个attention layer，与e_i进行一番计算（GRU的计算方式）后得到e_i+1，最终的e_n接一个softmax进行分类。  
这篇文章还在词向量中做了手脚，尝试了随机初始化并训练词向量、使用预训练词向量并微调、使用预训练词向量三种方法，让我有点意外的是直接使用预训练词向量不微调比微调效果还要好，而且差距不小。作者认为预训练词向量中，不同词之间存在相似性，但是微调破坏了这种相似性；另一方面，训练集中的词得到了微调，测试集中的OOV词汇却没有得到微调，也破坏了这种相似性，并导致了过拟合。  
此外结论处作者还表示```we need a mechanism to stop the attention process automatically if no more useful information can be read from the  emory. We may also try other memory weighting strategies to distinguish multiple targets in one comment more clearly.```  
Acc: Restaurant80.23%, Laptop74.49%, Tweet69.36%.   
Macro-F1: Restaurant70.80%, Laptop71.35%, Tweet67.30%.  

9. IJCAI2017_Interactive Attention Networks for Aspect-Level Sentiment Classification
这篇文章我总感觉想法很好，最后做起来感觉怪怪的。
之前的论文都是将context喂如LSTM得到中间状态，target词直接使用词向量（多个词则取平均），然后计算context的attention，最后用来分类。这篇文章觉得前面的人都只考虑target影响到context中不同词的权重，但是context反过来也会影响到target，而且target可能是多个词构成的词组（想一下的话是非常常见的，因为这里的target实际上是指entity+aspect了），那其实target中的每个词的权重应该也是不一样的。因此这篇文章将target和context分别喂入LSTM中得到隐层状态，分别对两边的隐层状态进行average-pooling操作得到一个向量，用该向量分别计算另一边的attention，最终将两边的输出拼到一起用softmax分类。
attention的计算方式采用了```AAAI2017_Attention-Based LSTM for Target-Dependent Sentiment Classification```中提到的双线性项的方式。
Acc: Restaurant78.6%, Laptop72.1%(被各种2016的会议摁着摩擦...)