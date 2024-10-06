## 第一节
#### basic kl
![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726709815099-d4d2af76-1517-403a-81c9-29a9b5293f59.png)

> ### 1. Dot product（内积）比较词语的相似性：
> + ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726711594899-cbaaef0a-e7ec-4798-920a-4c9e8ca73c05.png)
>
> ### 2. 指数函数使一切都变为正数：
> + 由于点积的值可能为负数，但概率必须是非负的，所以通过指数函数 exp⁡可以确保输出是正数。这个值越大，表示该词与上下文的匹配程度越高。
>
> ### 3. 归一化生成概率分布：
> + ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726712066547-df77bcac-02fc-490a-99a8-ee927fffa9e7.png)
>
> ### 公式解释：
> + ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726712166429-2d505907-2cec-4b79-ad2f-c3424388b23e.png)
> + **上下文词 c**：在自然语言处理模型中，比如 word2vec，训练时会使用词的上下文关系。上下文词 c 是你已经知道的词，它是目标词 o 周围的词（在给定句子或文本窗口中），模型利用它来预测目标词 o 的概率。
> + **目标词 o**：这是模型需要预测的词，给定上下文词 c 后，模型通过计算来估计目标词 o出现的概率。
>



:::tips
+ softmax函数：将向量转换为0-1的概率![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726709874767-2724989c-debc-4031-b3f9-882fa579a12d.png)

#### 词表示：
+ 机器了解了词的意思，就可以拿该词和其他词做相似度计算

#### nlp任务：
+ part of speech词性标注，标出是动词还是名词
+ co-reference：共指消解，识别代词

#### language model：根据前面的词预测后面的词的任务
##### 神经元：
![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726790571364-a152eea4-00e9-4ef3-aa25-8fbcbdaa1412.png)

##### 为什么需要引入非线性激活函数：
![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726790888404-614ebc4e-893f-40ec-b4a3-edd401c9900d.png)

_防止多层神经网络坍塌成单层神经网络(即无论堆叠多少层都只是一个线性函数)，如图，多层神经网络可以转换成单层神经网络，表达能力一致_

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726791016841-d94acd80-d3f2-4d25-8b26-eda31c0d6f72.png)

+  简单的线性模型无法解决 XOR 问题  
+ **表达能力**（Expressive Power）指的是神经网络**表示复杂函数的能力**
+ **sigmoid**解决二分类问题，将输出压缩到0-1之间作为预测某个类别的概率
+ **softmax**解决多分类问题，![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726791659057-c1f1d2c3-9eb9-498c-9172-08758eff1cd5.png)，保证是正值，同时总和为1

:::

### <font style="color:rgb(79, 79, 79);">1.1传统的NLP方法-使用词典或词库[</font><font style="color:rgb(0, 0, 0);background-color:rgb(248, 248, 64);">WordNet</font><font style="color:rgb(79, 79, 79);">]：</font>
+ <font style="color:rgb(79, 79, 79);">WordNet中包含了同义词和上位词，动物是狗的上位词</font>

**WordNet的缺点：**

+ 同义词之间也有细微的差别，体现不出来：<font style="color:rgb(77, 77, 77);">硕大和大，在某个网络中可能被列为同义词，但我们不能在“大梦一场”或者“一件大事”中使用“硕大”来替换“大“</font>
+ 语料库不能实时更新，除非消耗巨大的人力资源
+ 无法准确计算词语之间的相似度：悲伤和难过的相似度应该是多少，人为定义太过主观，不同的场合有不同的定义

### 1.2<font style="color:rgb(79, 79, 79);">传统的NLP方法-用离散符号表示词[</font><font style="color:rgb(0, 0, 0);background-color:rgb(248, 248, 64);">one-hot向量</font><font style="color:rgb(79, 79, 79);">]</font>
+ 将词看成一个离散的符号，每个词便可以被表示为一个one-hot向量(只有一个位置是1，其余位置都是0)

**独热编码的缺点：**

+ 词数过多，则向量维度过大
+ 两个one-hot向量是正交的，无法计算词之间的相似性

### <font style="color:rgb(79, 79, 79);">1.3 现代的统计学NLP方法—使用上下文表示词语[</font><font style="color:rgb(0, 0, 0);background-color:rgb(248, 248, 64);">词向量</font><font style="color:rgb(79, 79, 79);">]</font>_<font style="color:rgb(79, 79, 79);"></font>_
_**词向量也被称为词嵌入（word embedding）or 词表示（word representation），**_它们都是一个**分布式表示**

+ 分布时语义：一个词是由其附近的词确定的
+ 如何实现完形填空：**在高维空间里为每个词找一个位置，如图**![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726835470622-c849ecbd-86bf-48d2-bf67-84d5da00fafc.png)
+ 接下来用一个密集的向量表示每一个词，通过向量点积衡量词之间的相似性，相似的词向量相似

**怎么学习得到词向量：Word2Vec算法**

+ **Word2Vec模型如何实现，算法思路**：
    - 有一个语料库，里面有很多词句
    - 每个单词用一个向量表示（实际算法一般是两个向量，一个中心词向量，一个是周边词向量），每个单词的向量维度是一致的，向量里的值是随机赋予的随机数
    - 以位置t为索引(从0个词开始到最后一个词)，从头到尾遍历这个语料库，获得一个这个位置的中心词汇c和中心词汇外部的上下文词汇集合o
    - 利用c和o的相似度来计算根据c得到o的概率，即c发生的条件下，o发生的概率<font style="color:rgba(0, 0, 0, 0.75);">P(o|c)</font>
    - <font style="color:rgba(0, 0, 0, 0.75);">调整词向量里的值来提高</font><font style="color:rgb(0, 0, 0);background-color:rgb(248, 248, 64);">P(o|c)，使得概率最大化</font>
    - <font style="color:rgb(0, 0, 0);">重复之前的步骤，知道概率p无法再提高</font>
+ **Word2Vec的主要步骤：**
    - 随机的词向量开始
    - 遍历整个语料库的词
    - 用词向量预测周围的词
    - 更新向量，不断训练
+ **Word2Vec的参数计算过程：**
    - 根据中心词V和外围词U做点积，再归一化，得到概率估计值，如图
    - ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726880444108-957e5a22-f283-4207-953a-3aada5ac20fa.png)
+ **<font style="color:rgb(77, 77, 77);">Word2vec 通过将相似的词放在临近的空间（见图 2.3）来最大化目标函数：</font>**
+ ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726880891271-48d3ca87-13f4-4cf7-8c5f-fdb49e992f75.png)
+ **<font style="color:rgb(77, 77, 77);">t-SNE</font>**<font style="color:rgb(77, 77, 77);">(t-distributed Stochastic Neighbor Embedding)的思想就是</font><u><font style="color:rgb(77, 77, 77);">将高维的数据，通过</font></u>**<u><font style="color:rgb(77, 77, 77);">条件概率</font></u>**<u><font style="color:rgb(77, 77, 77);">来表示点与点之间的相似性，</font></u><font style="color:rgb(77, 77, 77);">t-SNE 是**</font><u><font style="color:rgb(77, 77, 77);">少数可以同时考虑数据全局与局部关系的算法</font></u><font style="color:rgb(77, 77, 77);">**，在很多聚类问题上效果很好</font>
+ **Word2Vec算法：两个模型，两种算法**
    - **两种模型：**CBOW和Skip-Gram，前者根据周围词预测该词，后者根据中心词预测周围词
    - **两种算法：**<font style="color:rgb(77, 77, 77);">Negative Sampling 和 Hierarchical Softmax，Negative Sampling 通过抽取负样本来定义目标；H ierarchical Softmax 通过使用一个有效的树结构来计算所有词的概率来定义目标。</font>



<font style="color:rgb(85, 86, 102);background-color:#D8DAD9;">softmax函数简而言之就是</font>**<font style="color:rgb(85, 86, 102);background-color:#D8DAD9;">将函数的输出映射到[0, 1]</font>**<font style="color:rgb(85, 86, 102);background-color:#D8DAD9;">。并且通过指数函数的方式，放大了分布中最大分子的概率，同时也保证更小的分子项有一个大于0的概率，并且整个函数是可导的：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726837654411-794cbd86-af4b-492c-a12e-7ceda21fa11d.png)

+ _每个词都创建两个向量：该词语作为中心词时的向量v，作为上下文词时的向量u：_
+ ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726878742388-1de2e82a-d8e4-4e6a-9cee-8183f22fbe3c.png)
+ ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726878797853-de890d86-4758-463d-85dc-d3eaa62a5a30.png)



_**<u>Word2Vec的数学推导：</u>**_







---

## 第二节
### <font style="color:rgb(79, 79, 79);">本篇内容覆盖</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">算法优化基础</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">计数与共现矩阵</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">GloVe模型</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">词向量评估</font>
+ <font style="color:rgba(0, 0, 0, 0.75);">word senses</font>



### 共现矩阵
构建词向量的一种思路是借助于共现矩阵（设为X），可以基于窗口或全文档（full document）统计

构建共现矩阵的两个条件：

+ Window：在每个单词周围用窗口捕获信息
+ 文档Word-document

### 共线向量：
一个共线向量是共现矩阵的一行，代表一个词向量，直接基于共现矩阵构建词向量，会出现下列问题：

+ 使用共现词数衡量单词相似性，向量随着词汇量的增加而增加
+ 后续分类模型存在稀疏问题，导致模型不太稳健

**解决方案：**

_<u>构建低维向量：</u>_将重要信息存储在固定的少量维度，构建密集向量

**降维方法：奇异值分解SVD**

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726882399758-331607a7-31a4-431a-92fb-d078984463fb.png)

**k-秩近似**

k-秩近似是降维的一个概念，<font style="color:rgb(77, 77, 77);">*矩阵的秩(rank)*是矩阵中线性无关的行(或列)向量的最大数量。</font>

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726882982200-11c2e1ae-9525-49ff-a435-1c497fe5ce4b.png)

**<font style="color:rgb(77, 77, 77);">矩阵的秩</font>**<font style="color:rgb(77, 77, 77);">可以被认为是由矩阵表示的</font>**<font style="color:rgb(77, 77, 77);">独特信息量多少的代表</font>**<font style="color:rgb(77, 77, 77);">。秩越高，信息越高。</font>

### <font style="color:rgb(77, 77, 77);">GloVe</font>
GloVe：[bal](https://nlp.stanford.edu/projects/glove/)[Ve](https://nlp.stanford.edu/projects/glove/)[ctors for Word Representation](https://nlp.stanford.edu/projects/glove/)

<font style="color:rgb(31, 35, 40);">流程：输入语料库--> 统计共现矩阵--> 训练词向量-->输出词向量</font>

<font style="color:rgb(31, 35, 40);"></font>



---

## w_nlp
#### feature extraction
+ **稀疏表示：**用较少的非零值表示 ![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726885367173-d832f26a-71e3-475f-8269-d2508e74e246.png)

> 图示：如何将文本表示为向量
>

#### negative and positive
![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1726885653640-7334e7c6-3dc8-4f96-9a9b-bab4dc5ebe43.png)

> 数字是该单词出现的次数
>

## thu_nlp
### Word2Vec
Word2Vec有两种模型：_**continuous bag-of-words 和 continuous skip-gram**_

<u>bag-of-words：不考虑单词顺序，每次输出结果都一样</u>

word2vec利用**滑动窗口**构造训练数据，在一个窗口里，最中间的词叫target，其他词时context

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727224159624-5772bcaa-7db9-445d-b359-d81f1138f692.png)

> 左边是CBOW，根据context预测target，右边是skip-gram，根据target预测context
>

### RNN
**rnn特点：**

+ 处理序列数据时会顺序记忆（即人类看错词）
+ 参数共享![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727225435131-9c0ddb43-bc40-4a7a-a23c-a534444ab02a.png)

**rnn缺点：**

+ 时间慢，需要计算前一个单元的结果
+ 序列太长，会丢失前面的信息
+ 会出现梯度消失或爆炸，所以出现变体解决问题

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727225250470-faa85d3e-c784-41ed-b9c9-6cbff83f7a27.png)

#### GRU
GRU:门控循环单元

:::tips
+ 门控机制：对输入信息筛选
+ GRU会有两个门：更新门和重置门，作用是权衡过去时间步的信息和当前输入的比重
+ 每个门会有自己专属的权重![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727225932418-6d61d3f9-c33b-40c1-8ec1-0888593cc179.png)
+ 重置门：考虑上一层的隐层状态，对当前的激活通过计算获得一个新激活
+ 更新门：权衡得到的新激活和上一个时间步的状态之间的影响

:::

#### LSTM
_<u>LSTM：长短时记忆网络</u>_

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727226772583-f33bdac5-3947-44db-9679-0d32a51fc3c3.png)

> lstm增加了一个cell state，即细胞状态，用于学习长期的依赖关系
>



![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727228048339-44de1d9f-9e4d-4ac3-94c3-653276ee73f6.png)

> **遗忘门：**通过这个公式计算得到一个0-1之间的值，如果为0，则代表过去的信息直接丢弃
>

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727228297627-b137a861-c404-431c-8a2c-1c50125b1efc.png)

> **输入门：**决定Ct中哪些信息存储到cell state中
>

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727228567414-aa551945-da54-4aa5-b164-822a3afaddef.png)

> 遗忘门的数值*上一个细胞状态，决定哪些信息保留丢弃忘记，
>

![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727228743418-d5624a7c-3f20-4160-963b-5a01e117641a.png)

> **输出门：**
>

[cs231n](https://www.yuque.com/yuqueyonghuftqtcn/th8glr/yqo2mwctwkp8ed5z)

#### 双向RNN
普通RNN根据过去状态和当前输入获取信息

双向RNN：依赖未来的信息或者说整个序列的信息

### 预训练语言模型PLM
语言模型：根据前面预测后面

预训练语言模型（bert，gpt，word2vec等）可以拿学到的知识迁移到下游任务中

**预训练语言模型分为两种：**

+ feature-based ：模型的参数在下游任务中保持不变，word2vec
+ fine-tuning：参数通过下游任务的训练实时更新，bert，gpt

:::tips
+ 双向rnn：会信息泄露，模型会学到shortcut
+ bert：解决了信息泄露问题，但是预训练时有mask token，下游任务时没有，会导致预训练和fine-tuning的差异，导致模型效果变差

:::

## MLM（masked LM）
MLM是bert核心的预训练任务

## Bert
● pre-training：预训练，在一个数据集上预训练一个模型，再用到别的任务(training)上
● sentence-level：句子和句子之间
● token-level：词和词之间
● 自注意力机制没有可学习参数，但是多头注意力会把k,v,q分别做一次投影
● 图片增强为了防止过拟合
● 普通语言模型：根据上文预测下文
● wordpiece：如果一个词出现概率不大，就把它切开看子序列（词根），取其中经常出现的子序列即可

BERT特性
cls（classification）：用来表示分类，和其他所有token做交互，获取的是整个序列的信息
● bert训练时有mask，微调时没有

bert序列的第一个词永远是cls，sep来区分两个句子
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709606394573-a21439c3-0e32-44aa-9cfc-e3d0162cac02.png?x-oss-process=image%2Fformat%2Cwebp
bert减轻了之前语言模型的单向限制，使用了一个MLM(masked language model，带掩码的语言模型)，即训练了一个双向的transformer，所以bert不能做机器翻译，不适合生成任务
BERT用transformer做编码器
bert的预训练：在没有标注的数据上训练
bert的微调：权重初始化成预训练中间得到的权重，所有的权重在微调时都会参与训练，用的是标注的数据

它是一种基于双向Transformer编码器进行预训练的语言模型，能够有效地捕捉词语和句子级别的representation。BERT模型的创新之处在于其预训练方法和任务设计，其中包括使用Masked Language Model（MLM）和Next Sentence Prediction（NSP）两种方法来捕捉词语和句子级别的representation。
BERT模型的特点包括：
1. 双向Transformer编码器：BERT利用双向Transformer编码器来进行预训练，这比传统的单向RNN或LSTM模型更有效，因为它可以同时考虑单词前后的上下文信息。
2. 预训练方法：BERT通过结合Masked LM和NSP两种预训练任务来实现高效的学习。在Masked MLM任务中，模型需要预测被遮挡的单词，而在NSP任务中，模型需要判断给定的两个句子是否连续。

## transformer

https://cdn.nlark.com/yuque/0/2024/png/34701129/1711262089931-bb49dd83-9d8a-4fc6-bb65-2f0ae100fc75.png?x-oss-process=image%2Fformat%2Cwebp

● rnn难以并行计算，序列变长，前面的历史信息可能会丢掉
● 自回归：过去时刻的输出成为当前时刻的输入
● 不同的注意力机制算法不同
● 权重等价于query和key的相似度，输出就是value的加权和 
● additive attention：加型注意力，处理key和query不等长的情况
● 注意力机制一次能看到全部输入
● 输入的词就是token
● attention与序列信息无关

transformer是第一个完全基于注意力机制来做encoder到decoder的架构的模型
矩阵乘法很好做并行
#### transformer特点
● 摒弃了CNN和RNN,使用了attention注意力机制，自动捕捉输入序列不同位置的相对关联，擅长处理长文本，可以高度并行，训练速度快
自注意力机制
即 key,value,query是同一个，就是自己本身

编码器：
输出就是输入的加权和，权重来自于自己本身和其他向量的相似度(无多头和投影时），权重即query和key之间的距离
 transformer的编码器和解码器都是自注意力机制
输入的序列，每一个词就是一个position
#### embedding
即对于输入的token，对于任何一个词，将其映射成对应的向量
给定一句话，把顺序打乱，atttention出来的结果一样
即attention不包含时序信息，但rnn包含(上一个时刻的输出作为下一个时刻的输入)
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709052074174-a5dca84e-f402-4712-aa5f-c020fa533037.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_610%2Climit_0

编码器：对第i个元素抽特征时，可以看到整个序列里面的所有元素，解码器：掩码使得不能看到i后面的元素
注意力机制
#### 注意力机制
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709052993733-611ce7dc-4b52-4539-bf87-a059636c1336.png?x-oss-process=image%2Fformat%2Cwebp
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709052993733-611ce7dc-4b52-4539-bf87-a059636c1336.png?x-oss-process=image%2Fformat%2Cwebp
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709053025695-6150f2d8-64b6-414b-8515-3ccdbfdccc04.png?x-oss-process=image%2Fformat%2Cwebp
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709271233630-2a870f8c-815c-4255-9dbb-f252f0cbbf41.png?x-oss-process=image%2Fformat%2Cwebp

每一个value的权重是通过value对应的key和query的相似度算来的
https://cdn.nlark.com/yuque/0/2024/png/34701129/1709273011891-08764cb8-f76a-4029-84d2-dcf6e780031a.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
masked使得该时刻之后的输出的权重为0 
#### 多头注意力
https://cdn.nlark.com/yuque/0/2024/png/34701129/1710289722849-903a707d-ebe1-4562-b0fe-b776146bdaa7.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
key，value，query都是一个长为d的向量，通过一个全连接层映射到低维
https://cdn.nlark.com/yuque/0/2024/png/34701129/1710290156848-4330bd3d-5157-4d63-ae85-549318485fb7.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

## transformer搭建
https://cdn.nlark.com/yuque/0/2024/png/34701129/1711262501103-02860d05-b711-4084-bfd8-386db816b107.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_918%2Climit_0

https://cdn.nlark.com/yuque/0/2024/png/34701129/1711262593663-a59fb201-6c62-4736-b6e3-5a7192d5628f.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

decoder是不能并行的，因为要根据之前的输出预测输入，为了解决引入了teacher forcing：将整个标签全部输入，然后mask

https://cdn.nlark.com/yuque/0/2024/png/34701129/1711265898777-43e6b86f-a550-48ab-9a90-8465bbe127c2.png?x-oss-process=image%2Fformat%2Cwebp

https://cdn.nlark.com/yuque/0/2024/png/34701129/1711266338962-d5f34676-ac4c-465d-85dd-4b95bea35f1c.png?x-oss-process=image%2Fformat%2Cwebp

https://cdn.nlark.com/yuque/0/2024/png/34701129/1711266467540-46c34834-c887-4a83-9d2d-bf44575b96f3.png?x-oss-process=image%2Fformat%2Cwebp


### 阿里云transformer搭建
模型输入：embedding和positional embedding
##### embedding层
● 将输入数据变成模型可以处理的向量，描述原始数据所包含的信息
● embedding层的输出可以是：word embedding（文本任务）

import math
import torch
import torch.nn as nn

# 构建embedding层
class Embeddings(nn.Module):
    '''
    d_model:word embedding的维度
    vocab:词表的大小
    '''
    def __init__(self,d_model,vocab):
        super(Embeddings, self).__init__()
        # lut:lookup table
        # 获得一个词嵌入对象self.lut
        self.lut=nn.Embedding(vocab,d_model)
        self.d_model=d_model

    def forward(self,x):
        '''
        x:代表输入给模型的单词文本通过词表映射后的one-hot向量
        '''
        embedds=self.lut(x)
        return embedds*math.sqrt(self.d_model)
##### 位置编码
transformer是同时输入，并行推理，所以缺失了位置信息
位置编码可以是固定的，也可以是可学习的参数
最终模型的输入是若干个时刻对应的embedding，每个时刻对应一个embedding，既包含了本身的语义信息，也包含了当前时刻在整个句子中的位置信息

这里采用固定的位置编码：
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712749248758-6a596b6c-f7bf-419c-8c4f-4cd363c28c4d.png?x-oss-process=image%2Fformat%2Cwebp
位置编码长度=embedding层，设置为512

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout,max_len=5000):
        super(PositionalEncoding, self).__init__()
        # max_len：每个句子的最大长度
        self.dropout=nn.Dropout(p=dropout)

        # 计算位置编码,这里的计算方式和公式不同但等价
        # 这样计算是为了避免中间的数值计算结果超出float的范围
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2))*-(math.log(10000.0)/d_model)
        # [:,0::2]选中所有行，对于每一行中的元素，索引从0开始，步长为2
        # 即返回每一行中偶数索引的元素
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        '''
        将tensor注册为模型的缓冲区(buffer)，即该张量不会被视为参数
        不需要计算梯度，不会在反向传播中更新
        '''
        self.register_buffer('pe',pe)
    def forward(self,x):
        x=x+Variable(self.pe[:,:x.size(1)],requires_grad=False)
        return self.dropout(x)
##### encoder
推理时encoder只推理一次，decoder类似rnn不断循环推理，生成预测结果
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712753640512-ebe11f6e-1045-4b50-ad80-c3ca1a3d38b0.png?x-oss-process=image%2Fformat%2Cwebp
最开始的时候，将编码器提取的特征以及一个句子起始符传给decoder，decoder会输出第一个单词I，然后将第一个单词I输入给decoder，再预测下一个单词love,再将 I love喂给decoder
encoder的作用：对输入进行特征提取，为解码器提供语义信息
https://cdn.nlark.com/yuque/0/2024/webp/34701129/1712991201092-84f1dc0d-2245-4784-8bfc-c8c43d9e409c.webp?x-oss-process=image%2Fresize%2Cw_713%2Climit_0
注意transformer encoder decoder是自注意力
# encoder
# 定义一个clones函数，便于将某个结构复制n份
def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self,layer,N):
        super(Encoder, self).__init__()
        # 将layer堆叠6层
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,mask):
        for layer in self.layers:
            x=layer(x,mask)
        return self.norm(x)

# 残差连接实现
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout):
        super(SublayerConnection, self).__init__()
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        # 论文的方案
        # sublayer_out=sublayer(x)
        # x_norm=self.norm(x+self.dropout(sublayer_out))

        # 调整后的版本
        sublayer_out=sublayer(x)
        sublayer_out=self.dropout(sublayer_out)
        x_norm=x+self.norm(sublayer_out)
        return x_norm

class EncoderLayer(nn.Module):
    # 两个sublayer：self-attention和feed forward
    def __init__(self,size,self_attn,feed_forward,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn=self_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),2)
        self.size=size

    def forward(self,x,mask):
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,mask))
        z=self.sublayer[1](x,self.feed_forward)
        return z
##### 注意力机制
**注意力计算**：需要三个输入qkv，通过公式得到注意力的计算结果
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712758077903-34f18f01-072d-4880-ab00-8b3c4b82bf75.png?x-oss-process=image%2Fformat%2Cwebp
attention score：softmax（）这部分
为的维度大小。这个除法被称为Scale。当很大时，的乘法结果方差变大，进行Scale可以使方差变小，训练时梯度更新更稳定。
即scale用于防止点积结果过大，避免梯度消失或爆炸
计算流程图：
https://cdn.nlark.com/yuque/0/2024/webp/34701129/1712758156152-f9447409-d9de-4c96-9e99-706003022859.webp?x-oss-process=image%2Fresize%2Cw_278%2Climit_0
当前时刻的注意力计算结果，是value的加权和
权重：query和key做内积得到相似度
# 注意力机制
def attention(q,k,v,mask=None,dropout=None):
    # 取query最后一维的大小，对应词嵌入维度
    d_k=q.size(-1)
    # 按照注意力公式，将query和key的转置相乘
    # 这里key将最后两个维度转置，再除以缩放系数得到注意力得分张量scores
    scores=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)

    # 判断是否使用掩码张量
    if mask is not None:
        scores=scores.masked_fill(mask==0,-1e9)

    # 对scores最后一维进行softmax，得到最终的注意力张量
    p_attn=F.softmax(scores,dim=-1)

    # 判断是否使用dropout
    if dropout is not None:
        p_attn=dropout(p_attn)
    return torch.matmul(p_attn,v),p_attn
##### 多头注意力
不同的头可以关注到同一个词不同的语义，比如bank：银行、河岸
# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self,num_heads,d_model,dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        # 判断num_heads能否被d_model整除，因为需要给每个头分配等量的词特征
        # 即word embedding/head
        assert d_model%num_heads==0
        # 得到每个头的word embedding维度
        self.d_k=d_model//num_heads
        self.num_heads=num_heads
        # 通过linear，即w矩阵得到qkv,还有最后拼接的wo矩阵
        self.linears=clones(nn.Linear(d_model,d_model),4)
        # self.attn代表最后得到的注意力张量，现在还没有结果所以为None
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,q,k,v,mask=None):
        if mask is not None:
            mask=mask.unsqueeze(1)
            # 获得一个变量，代表有多少样本
            nbatches=q.size(0)

            # 这里transpose是为了让句子长度维度和词向量维度相邻
            # 注意力机制才能找到语义和句子位置的关系
            # 得到每个头的输入qkv
            q,k,v=\
            [l(x).view(nbatches,-1,self.num_heads,self.d_k).transpose(1,2)
             for l,x in zip(self.linears,(q,k,v))]

            x,self.attn=attention(q,k,v,mask=mask,dropout=self.dropout)

            # 得到每个头计算的结果组成的4维张量，需要将其转换为输入的形状方便后续计算
            x=x.transpose(1,2).contiguous()\
                .view(nbatches,-1,self.d_k*self.num_heads)
            # 使用linears中的最后一个线性变换（wo矩阵）得到最终的多头注意力的输出
            return self.linears[-1](x)
##### 前馈全连接层
包含两个线性变换和一个ReLU
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712765690061-d95c9652-06e5-445d-a555-623f362e80d1.png?x-oss-process=image%2Fformat%2Cwebp
attention模块中每个时刻的输出都整合了所有时刻的信息
但是ffn每个时刻与其他时刻的信息无关
# 前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # 希望通过ffn全连接层后输出和输入维度一致，d_ff就是第二个linear的输入
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
##### norm层
# Norm层:将数值规范在合理范围内
class LayerNorm(nn.Module):
    def __init__(self,feature_size,eps=1e-6):
        # feature_size：词嵌入的维度
        # eps：足够小的数，防止归一化公式分母为0，默认为1e-6
        super(LayerNorm, self).__init__()
        self.a_2=nn.Parameter(torch.ones(feature_size))
        self.b_2=nn.Parameter(torch.zeros(feature_size))
        self.eps=eps

    def forward(self,x):
        # x来自上一层的输出，首先对x求最后一个维度的均值，并保持输入输出维度一致
        # 接着再求最后一个维度的标准差
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2
##### 掩码
掩码：一般只有0和1，代表遮掩和不遮掩
掩码的作用：
● encoder中的掩码：屏蔽掉无效的padding区域
● decoder中的掩码：屏蔽掉来自未来的信息+屏蔽掉无效的padding区域
屏蔽掉无效的padding区域：一个batch中不同样本的输入长度不同，需要设置一个max_length，然后对空白区域填充，但是填充的区域在计算时无意义，所以需要mask掉
屏蔽掉未来信息：attention会获取所有时刻的信息，需要屏蔽掉未来信息
0：mask的位置，1：保留的位置
掩码通常设置为上三角矩阵，其中所有对角线以下的元素都是0，以确保模型在预测时不会接收到未来的信息
# 生成屏蔽未来信息的mask掩码张量：attention mask
# size是掩码张量最后两个维度的大小
def subsequent_mask(size):
    attn_shape=(1,size,size)
    # 用np.ones向这个shape中添加1，形成上三角矩阵
    '''
    ones函数创建一个形状为attn_shape的全1数组。
    triu函数将这个数组转换为上三角矩阵，其中对角线上方的元素为1，其余元素为0，
    astype函数将结果数组的数据类型转换为无符号8位整数（uint8）。
    subsequent_mask就是一个形状为attn_shape的上三角矩阵，
    '''
    subsequent_mask=np.triu(np.ones(attn_shape),k=1).astype('uint8')
    # 将numpy转换成tensor
    return torch.from_numpy(subsequent_mask)==0
##### decoder
解码器作用：根据编码器结果及上一次预测结果，预测下一个结果
解码器也是n个相同layer堆叠
细节：
● masked multi-head attention和编码器中的完全一致
● 第二个多头注意力中，q来自上一个子层，k和v来自编码器的输出
# decoder
class Decoder(nn.Module):
    def __init__(self,layer,N):
        super(Decoder, self).__init__()
        self.layers=clones(layer,N)
        self.norm=LayerNorm(layer.size)

    def forward(self,x,memory,src_mask,tgt_mask):
        # memory：编码器的输出 ，后两个分别代表源数据和目标数据的掩码张量
        for layer in self.layers:
            x=layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
        '''
        :param size: 词嵌入的维度大小，同时也代表解码器的尺寸
        :param self_attn: 多头自注意力
        :param src_attn: 多头注意力
        '''
        super(DecoderLayer, self).__init__()
        self.size=size
        self.self_attn=self_attn
        self.src_attn=src_attn
        self.feed_forward=feed_forward
        self.sublayer=clones(SublayerConnection(size,dropout),3)

    def forward(self,x,memory,src_mask,tgt_mask):
        '''
        :param x:上一层的输入
        :param memory:编码器的输出，存储了语义信息
        :param src_mask:源数据掩码张量
        :param tgt_mask:目标数据掩码张量
        
        '''
        m=memory
        # x传入第一个子层,自注意力，所以q,k,v=x,此时mask是为了屏蔽未来信息
        x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask))
        # 第二个子层,普通注意力机制,此时mask是为了屏蔽掉无意义的padding
        x=self.sublayer[1](x,lambda x:self.src_attn(x,m,m,src_mask))
        # 最后一个层，FFN
        return self.sublayer[2](x,self.feed_forward)


##### 模型输出
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712796745059-ac826651-4268-4187-91e1-087b4a66bbb6.png?x-oss-process=image%2Fformat%2Cwebp
linear：线性变换，转换维度，转换后的维度对应着输出类别的个数，如果是翻译任务，就对应的是字典的大小
# 模型输出
class Generator(nn.Module):
    def __init__(self,d_model,vocab):
        super(Generator, self).__init__()
        self.proj=nn.Linear(d_model,vocab)

    def forward(self,x):
        return F.log_softmax(self.proj(x),dim=-1)
##### 结构搭建
# 模型构建
class EncoderDecoder(nn.Module):
    def __init__(self,encoder,decoder,src_embed,tgt_embed,generator):
        super(EncoderDecoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.src_embed=src_embed
        self.tgt_embed=tgt_embed
        self.generator=generator

    def forward(self,src,tgt,src_mask,tgt_mask):
        # src：源数据，tgt：目标数据
        memory=self.encode(src,src_mask)
        res=self.decode(memory,src_mask,tgt,tgt_mask)
        return res

    def encode(self,src,src_mask):
        src_embedds=self.src_embed(src)
        return self.encoder(src_embedds,src_mask)

    def decode(self,memory,src_mask,tgt,tgt_mask):
        tgt_embedds=self.tgt_embed(tgt)
        return self.decoder(tgt_embedds,memory,src_mask,tgt_mask)

def make_model(src_vocab,tgt_vocab,N=6,d_model=512,d_ff=2048,num_heads=8,dropout=0.1):
    '''
    N:编码器和解码器堆叠的次数
    d_ff:FFN中embedding的维度，默认2048
    '''
    c=copy.deepcopy
    attn=MultiHeadAttention(nun_heads,d_model)
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)
    position=PositionalEncoding(d_model,dropout)
    model=EncoderDecoder(
        Encoder(EncoderLayer(d_model,c(attn),c(ff),dropout),N),
        Decoder(DecoderLayer(d_model,c(attn),c(attn),c(ff),dropout)),
        nn.Sequential(Embeddings(d_model,src_vocab),c(position)),
        nn.Sequential(Embeddings(d_model,tgt_vocab),c(position)),
        Generator(d_model,tgt_vocab)
    )
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)
    return model
### 李宏毅transformer
##### self-attention
rnn输入部分：
把一个句子中的每个词都表示成一个向量，方法：one-hot encoding or word embedding
one-hot encoding的缺点：不包含语义信息，看不到类别之间的联系，如这里的狗和猫都是动物
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712904226918-96c9a495-01d4-4e36-bf60-801402aaeb3b.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
每个句子就是一个不同长度的向量的集合
word embedding：给每一个词一个包含语义信息的向量

rnn输出部分：
第一种输出：输入n个向量就输出n个label，如pos tagging：词性标注，标注每个词是动词还是名词
第二种输出：输入一个序列，只输出一个label，如文本情感判断
第三种输出：机器自己决定输出多少个label，如seq2seq机器翻译

self-attention：
self-attention会将整个sequence输入，输入几个向量就输出几个向量，输出的向量考虑了整个句子的信息
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712904884388-b079d7a6-fc12-44e4-a584-103b330a1773.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712905161014-a3004f3d-b0ed-44fc-843b-992326ab1d33.png?x-oss-process=image%2Fformat%2Cwebp
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712905611767-99010813-0546-4a77-8251-21295e41021f.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712905533278-50005152-afdf-49b3-8d5e-70f6e4072905.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

记得还要自己和自己计算相似度
这里softmax可以被替换成别的激活函数
● 得到α'之后，要根据attention score抽取内容，即α分别*Wv矩阵得到v
● 然后让v和α'（attention score）相乘加权求和，即最终输出一个加权和
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712905898535-891de5cb-92a5-431b-bac2-e5128bd820a0.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906056770-90897abd-e514-49be-9743-5686515b58d3.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

输出向量b是同时计算出来的
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906344799-896d1cde-533f-4217-80f4-5006d265713f.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

attention score的计算：
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906544287-943eb86e-5c59-4b1f-b1e2-91d788343583.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906758301-81044956-3e8b-4997-bd4a-a0aada6bc608.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906835683-9262e280-8b9e-46cb-9a85-2d840004688e.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712906895341-c6da7f8f-d30f-452e-8199-62df2acc476c.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0


wqkv三个矩阵是可学习的参数
##### multi-head attention
head：超参数
多头注意力：一个词输入对应多个qkv，即多义词
q1之和k1算，不考虑k2，1对1,2对2
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712907230178-9ab55cc4-a99f-4981-9a99-bec1d1be40c5.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
positional encoding
位置编码可以是固定的(用公式计算)，也可以是learnable parameter
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712983423120-3a8e1a52-38b2-4485-84d2-08dfc457f047.png?x-oss-process=image%2Fformat%2Cwebp
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712983463977-9fd52c8a-3b58-45a1-b727-9b02cb0f8fd7.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712983590344-75f6046f-804b-47c8-97aa-46c10700057e.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0


rnn无法并行处理，而self-attention可以同时输出，速度更快
##### transformer
transformer是一个seq2seq的model，主要处理输出和输入不等长的问题，如机器翻译
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712984868495-a157635f-91fd-4262-ac6d-1ba12deeadec.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_666%2Climit_0
BOS：begin of sentence
##### masked attention
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712985420616-d034798e-dd70-4667-ac71-bbff8f180e4c.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
q2只和k1和k2做计算
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712985679023-b08ca5e3-f5ea-48e7-80ac-7c6e25158e46.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
NAT：non-atuoregressive model
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712985834073-5f208005-b3a0-4aab-a33f-1a30e35b1ead.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
FC：fully connected Network
##### 交叉注意力
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712987461689-cbb1ec54-0a62-45e4-b047-9a8cfff38e44.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712987708731-181e0a38-0c86-4fce-ab01-985e37705d55.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_937%2Climit_0

即query来自于decoder，key和value来自于encoder
transformer之前就有了cross-attention，但没有self-attention
交叉注意力可以有好几种方式
https://cdn.nlark.com/yuque/0/2024/png/34701129/1712987905167-dbc9a907-41e9-4081-a1cd-1df14d90d2cc.png?x-oss-process=image%2Fformat%2Cwebp%2Fresize%2Cw_1200%2Climit_0


[BERT](https://www.yuque.com/yuqueyonghuftqtcn/th8glr/iftdmwgr7ia2zwrk)

[transformer](https://www.yuque.com/yuqueyonghuftqtcn/th8glr/cq3rqcp2tavghvre)

[transformer搭建](https://www.yuque.com/yuqueyonghuftqtcn/th8glr/vww6qloe659evnfr)



## 剧本角色情感识别
![](https://cdn.nlark.com/yuque/0/2024/png/34701129/1727030779323-e190fcc4-bb95-4fee-bab8-529505e86b7f.png)

```python
from tqdm import tqdm
import pandas as pd
import os

# functools.partial用于创建一个偏函数。偏函数是通过预设一个或多个参数值来固定一个函数，返回一个新的函数对象。
# 这个新函数对象可以接受剩余未预设的参数（和关键字参数）来进行调用。
from functools import partial
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertPreTrainedModel,BertTokenizer,BertConfig,BertModel
from functools import partial
from transformers import AdamW,get_linear_schedule_with_warmup

# 使用第四块gpu
os.environ['CUDA_VISIBLE_DEVICES']='3'

# 加载数据

# 只读模式打开文件，并指定文件编码为utf-8来支持多语言字符
with open('data/train_dataset_v2.tsv','r',encoding='utf-8')as handler:
    # .read():读取文件全部内容，按换行符分割，并去掉首行
    lines=handler.read().split('\n')[1:-1]

    data=list()
    # 遍历处理每行数据
    for line in tqdm(lines):
        sp=line.split('\t')
        # 即数据应该只有四个字段id，content，character，emotion
        if len(sp)!=4:
            print('Error:',sp)
            continue
        data.append(sp)

train=pd.DataFrame(data)
# 手动指定列名
train.columns=['id','content','character','emotions']

test=pd.read_csv('data/test_dataset.tsv',sep='\t')
submit=pd.read_csv('data/submit_example.tsv',sep='\t')
# 去掉emotions为空的数据
train=train[train['emotions']!='']

# 数据处理
train['text']=train['content'].astype(str)+'角色:'+train['character'].astype(str)
test['text']=test['content'].astype(str)+'角色：'+test['character'].astype(str)

# 将emotions中的元素转换为整数列表
train['emotions']=train['emotions'].apply(lambda x:[int(_i) for _i in x.split(',')])

train[['love','joy','fright','anger','fear','sorrow']]=train['emotions'].values.tolist()
test[['love','joy','fright','anger','fear','sorrow']]=[0,0,0,0,0,0]

# index：指定保存为csv文件时，是否保留行索引，默认pandas会将行号作为一列保存到csv文件中
# 设置为false则不会保留行索引
train.to_csv('data/train.csv',columns=['id','content','character','text','love','joy','fright','anger','fear','sorrow'],
             sep='\t',index=False)
test.to_csv('data/test.csv',columns=['id', 'content', 'character','text','love', 'joy', 'fright', 'anger', 'fear', 'sorrow'],
            sep='\t',
            index=False)



# 定义dataset
target_cols=['love','joy','fright','anger','fear','sorrow']
class RoleDataset(Dataset):
    def __init__(self,tokenizer,max_len,mode='train'):
        super().__init__()
        if mode=='train':
            self.data=pd.read_csv('data/train.csv', sep='\t')
        else:
            self.data=pd.read_csv('data/test.csv', sep='\t')

        self.texts=self.data['text'].tolist()
        self.labels=self.data[target_cols].to_dict('records')
        # tokenizer用于将原始的文本转换成模型可以输入的token序列
        self.tokenizer=tokenizer
        self.max_len=max_len

    def __getitem__(self, index):
        text=str(self.texts[index])
        label=self.labels[index]
        # self.tokenizer.encode_plus 是 Hugging Face Transformers 库中 Tokenizer 类的一个方法，
        # 用于对文本进行编码，返回一个字典
        encoding=self.tokenizer.encode_plus(text,
                                            # 添加特殊标记，cls、sep
                                            # cls:用于表示整个序列的信息，通常出现在输入文本的开头
                                            # sep：区分不同句子
                                            add_special_tokens=True,
                                            # 限制编码后序列的最大长度
                                            max_length=self.max_len,
                                            # 返回id，区分每个标记属于哪个句子
                                            return_token_type_ids=True,
                                            # 文本太短，则使用填充标记填充到最大长度max_len
                                            pad_to_max_length=True,
                                            # 返回注意力掩码，生成一个布尔张量，指示哪些标记是实际文本，哪些是填充的（false）
                                            return_attention_mask=True,
                                            # 将输出转换为PyTorch张量
                                            return_tensors='pt',
                                            )
        # 创建样本字典，attention_mask：注意力掩码，标记哪些token是填充，哪些是文本（1/true）
        sample={
            # 原始文本
            'texts':text,
            # input_ids:每个token特定的ID
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten()
        }

        for label_col in target_cols:
            sample[label_col]=torch.tensor(label[label_col],dtype=torch.int64)
        return sample

    def __len__(self):
        return len(self.texts)

# 创建dataloader
def create_dataloader(dataset,batch_size,mode='train'):
    shuffle=True if mode=='train' else False

    if mode=='train':
        data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    else:
        data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    return data_loader

# 加载预训练模型
PRE_TRAINED_MODEL_NAME='hfl/chinese-roberta-wwm-ext'
# 加载分词器
tokenizer=BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
base_model=BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

# 模型构建
class EmotionClassifier(nn.Module):
    def __init__(self,n_classes,bert):
        super(EmotionClassifier, self).__init__()
        self.bert=bert
        self.out_love=nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_joy=nn.Linear(self.bert.config.hidden_size,n_classes)
        self.out_fright = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_anger = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_fear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.out_sorrow = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self,input_ids,attention_mask):
        _,pooled_output=self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        love=self.out_love(pooled_output)
        joy = self.out_joy(pooled_output)
        fright = self.out_fright(pooled_output)
        anger = self.out_anger(pooled_output)
        fear = self.out_fear(pooled_output)
        sorrow = self.out_sorrow(pooled_output)

        return {
            'love': love, 'joy': joy, 'fright': fright,
            'anger': anger, 'fear': fear, 'sorrow': sorrow,
        }

# 参数配置
EPOCHS=1
weight_decay=0.005
data_path='data'
# 学习率预热，训练开始时，学习率会逐渐增加到指定的学习率，设置为0则不会预热，而是直接使用设定的学习率
warmup_proportion=0.0
batch_size=64
lr=2e-5
max_len=128

warm_up_ratio=0

trainset=RoleDataset(tokenizer,max_len,mode='train')
train_loader=create_dataloader(trainset,batch_size,mode='train')

valset=RoleDataset(tokenizer,max_len,mode='test')
val_loader=create_dataloader(valset,batch_size,mode='test')

model=EmotionClassifier(n_classes=4,bert=base_model)
# model.cuda()

optimizer=AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
total_steps=len(train_loader)*EPOCHS

scheduler=get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warm_up_ratio*total_steps,
    num_training_steps=total_steps
)

criterion=nn.CrossEntropyLoss()

# 模型训练
def do_train(model,data_loader,criterion,optimizer,scheduler,metric=None):
    model.train()
    global_step=0
    # 记录训练开始的时间
    tic_train=time.time()
    # 每一百次迭代记录一次日志
    log_steps=100
    for epoch in range(EPOCHS):
        losses=[]
        for step,sample in enumerate(train_loader):
            input_ids=sample['input_ids']
            attention_mask=sample['attention_mask']

            outputs=model(input_ids=input_ids,attention_mask=attention_mask)

            loss_love=criterion(outputs['love'],sample['love'])
            loss_joy=criterion(outputs['joy'],sample['joy'])
            loss_fright=criterion(outputs['fright'],sample['fright'])
            loss_anger = criterion(outputs['anger'], sample['anger'])
            loss_fear = criterion(outputs['fear'], sample['fear'])
            loss_sorrow = criterion(outputs['sorrow'], sample['sorrow'])
            loss = loss_love + loss_joy + loss_fright + loss_anger + loss_fear + loss_sorrow

            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step+=1

            if global_step%log_steps==0:
                print("global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s, lr: %.10f"
                      % (global_step, epoch, step, np.mean(losses), global_step / (time.time() - tic_train),
                         float(scheduler.get_last_lr()[0])))

do_train(model,train_loader,criterion,optimizer,scheduler)

# 模型评估
from collections import defaultdict

model.eval()

# 当访问不存在的键时，会自动创建一个键并初始化为指定的默认类型，这里的默认类型是list
# 即访问不存在的键时，defaultdict会创建一个空列表作为该键的值
test_pred=defaultdict(list)
for step,batch in tqdm(enumerate(val_loader)):
    b_input_ids=batch['input_ids']
    attention_mask=batch['attention_mask']
    logists=model(input_ids=b_input_ids,attention_mask=attention_mask)
    for col in target_cols:
        # 这里的索引就是对应的标签
        out2=torch.argmax(logists[col],axis=1)
        test_pred[col].append(out2.cpu().numpy())

    print(test_pred)
    break

# 模型预测
def predict(model,test_loader):
    val_loss=0
    test_pred=defaultdict(list)

    model.eval()
    for step,batch in tqdm(enumerate(test_loader)):
        b_input_ids=batch['input_ids']
        attention_mask=batch['attention_mask']
        with torch.no_grad():
            logists=model(input_ids=b_input_ids,attention_mask=attention_mask)
            for col in target_cols:
                out2=torch.argmax(logists[col],axis=1)
                test_pred[col].extend(out2.cpu().numpy().tolist())

    return test_pred

submit=pd.read_csv('data/submit_example.tsv',sep='\t')
test_pred=predict(model,valid_loader)

print(test_pred['love'][:10])
print(len(test_pred['love']))


label_preds=[]
for col in target_cols:
    preds=test_pred[col]
    label_preds.append(preds)

print(len(label_preds[0]))
sub=submit.copy()
sub['emotion']=np.stack(label_preds,axis=1).tolist()
# x是emotion列中的元素
sub['emotion']=sub['emotion'].apply(lambda x:','.join([str(i) for i in x]))
sub.to_csv('baseline_{}.tsv'.format(PRE_TRAINED_MODEL_NAME.split('/')[-1]),sep='\t',index=False)
sub.head()
```

