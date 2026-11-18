from attention_vocab_dataset import vocab, sentences
import torch
import torch.nn as nn
import math
unk=vocab["<UNK>"]
print(unk)
padding_num=vocab["<PAD>"]
print(padding_num)
complete_sentences=[]
#转换为id
def dealwithwords(sentences,complete_sentences):
    for sentence in sentences:
        temp_sentence = []
        for word in sentence:
            token_id=vocab.get(word, unk)
            temp_sentence.append(token_id)
        complete_sentences.append(temp_sentence)



dealwithwords(sentences[:],complete_sentences)
print(complete_sentences)

#def padding
def maxlenth(complete_sentences):
    temp_len = []
    for sentence in complete_sentences:
        length = len(sentence)
        temp_len.append(length)
    return max(temp_len)
max_len=maxlenth(complete_sentences)
print(max_len)

def padding(complete_sentences):
    for sentence in complete_sentences:
        if len(sentence)<max_len:
            for i in range(max_len-len(sentence)):
                sentence.append(padding_num)

padding(complete_sentences)
print(complete_sentences)
sen_tensor = torch.LongTensor(complete_sentences)
print(sen_tensor.size() )

#emmmmmmmmmmmmmmmmmmmmmmmmmmbedding

fea=6
'''
embedding=torch.randn(len(vocab),fea)

result_1=torch.zeros(sen_tensor.size(0),max_len,fea)
for i in range(sen_tensor.size()[0]):
    for j in range(sen_tensor.size()[1]):
        if sen_tensor[i][j]!=padding_num:
            num=sen_tensor[i][j]
            result_1[i][j]=embedding[num]
print(result_1)
        对应的向量的初始位置也是要训练的
'''



class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(fea, fea, bias=False)
        self.W_K = nn.Linear(fea, fea, bias=False)
        self.W_V = nn.Linear(fea, fea, bias=False)
        self.softmax=nn.Softmax(dim=-1)
        self.layer=nn.LayerNorm(fea)
        self.embedding = nn.Embedding(len(vocab), fea, padding_idx=0)


    def forward(self, sen_tensor):
        result = self.embedding(sen_tensor)
        X=result
        Q1=self.W_Q(X)
        K1=self.W_K(X)
        V1=self.W_V(X)
        point_1 = Q1 @ K1.transpose(-1, -2)
        point=point_1/math.sqrt(fea)
        pad_mask = (sen_tensor == padding_num).unsqueeze(1)
        point = point.masked_fill(pad_mask, float("-inf"))
        A=self.softmax(point)
        Attn=A@V1
        out=Attn+X
        finish=self.layer(out)
        return finish,A

model = Mymodel()
output, attn_weight = model(sen_tensor)
print("注意力权重形状:", attn_weight.shape)
print("输出特征形状:", output.shape)


