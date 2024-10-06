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

### MLM（masked LM）
MLM是bert核心的预训练任务



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

