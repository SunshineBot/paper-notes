# Natural Language Generation/Abstractive Summarization

## 调研目的：

了解生成式文本摘要的常用技术和当前的发展趋势，明确当前项目有什么样的摘要需求，判断现有技术能否用于满足当前的需求，进一步明确毕业设计方向及其可行性

## 调研方向：

- 项目中需要用到摘要的地方以及区别
- 数据集(研究用评测集/**项目用大规模数据集**)
- 现有技术
    - 分类
        - 有监督
        - 无监督
        - 半监督等（如果有）
    - 效果
    - 优势和缺点
- 评价现有技术用于当前项目的可行性
- *扩展*：寻找现有技术的研究改进方向

## 项目中用到摘要的地方

- 传统新闻摘要任务
    - 单/多文档新闻摘要生成
- 非传统摘要任务
    - 专家/机构观点标题生成
    - 评论标题生成
    - 特点
        - 篇幅一般较短
        - 不同位置的内容对摘要没有影响
        - 观点可能包含多种（受限于聚类效果），相当于噪声数据

## 评价方法

- 自动评价方法: Rouge
    - 基于N元模型，判断生成的摘要与参考摘要N元组重复比例
    - 自动评价方法本身也是被研究的对象
- 人工评价方法
    - 由人对摘要内容进行打分，包括可读性、综合质量等。

## 数据集

- LCSTS
    - 哈工大中文微博摘要数据集
    - 数据集内容
        - part1: 2.4m训练数据， （短文本，摘要）对
        - part2: 1w标注数据，给摘要和短文本的相关程度打分（1~5），用来去除part1中的噪声数据
        - part3: 1.1k对训练数据，独立于part1&2，由3人对摘要打分，一般保留3分以上的作为摘要训练数据
    - 数据量非常大，噪声非常大
- DUC2004/Gigaword
    - 抽取式摘要数据集
    - 单句话摘要
- CNN/Daily Mail
    - 生成式摘要数据集
    - 摘要包含多个句子，但是长度不是太长

## 思路

- Seq2seq + Attention(RNN->CNN)
- Pointer/Generation、CopyNet机制，以及其它的机制
- Extractor + Abstractor
- Reinforcement Learning
- GAN、unsupervised learning

<!-- ## Techniques -->
## General
- Category: text-to-text, data-to-text, image/video-to-text
- Tasks:
    - Content determination 确定生成内容
    - Text structuring 确定生成结构
    - Sentence aggregation 句子聚合
    - Lexicalisation 词法实现
    - Referring expression generation 指代生成
    - Linguistic realisation 语言实现
- Example:
    - ![snow](resources/textsum/snow.jpg)
    - 有一个穿红衣服的小孩子，在雪地里堆雪人。
- Example:
    - 高铁车票“无纸化”  
    近日，中国铁路总公司...  
    乘客或可实现“刷手机”、“刷身份证”直接进站乘车，而不需要在乘车之前特意换取纸质车票。...  
    最快今年四季度，中国铁路电子客票业务将开展试点运营。  
    ...
    - 最快今年四季度，乘客可直接刷手机或身份证直接进站乘坐高铁火车。

## Text-to-Text
- Document Summarization(abstractive)
    - Systems: NeATS, NewsBlaster, NewsInEssence, Summly
    - Evaluation: ROUGE
    - Tasks:
    - Category:
        - single/multi document summarization
    - Seq-to-Seq
        - attention mechanism
        - copying mechanism: 考虑到摘要中的很多字和原文相同，拷贝机制允许直接拷贝输入中的字作为输出，而不是总是通过隐层状态来生成字。
        - Reinforcement Learning: 直接通过Rouge来进行优化比decoder输出的结果的似然函数来优化效果更好
        - limit length
- Sentence Compression & Fusion
    - few researches
- Paraphrase Generation
    - few researches

## Data-to-Text
## Image/Video-to-Text
