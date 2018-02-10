
############
# 初始设置 #
###########
result_path='vec_new2.txt' # 结果位置
glove_path='glove.840B.300d.txt' # 字典位置
read_txt='all_f.txt' # 字符集合位置
embedding=300
featrue_embedding=64

#########################################################################
# 自己新建的word2vec库，针对原始文件中没有的单词训练，也可用来训练特征向量 #
#######################################################################
from gensim.models import word2vec
sentence=word2vec.Text8Corpus('f2.txt')
model=word2vec.Word2Vec(sentence,size=64,min_count=1)
# print(model.most_similar(['i']))
model.save('w2vf2model')

############
# 加载内容 #
###########
import nltk
nltk.download('punkt')

from gensim.models import word2vec
model=word2vec.Word2Vec.load('w2vmodel')
modelf1=word2vec.Word2Vec.load('w2vf1model')
modelf2=word2vec.Word2Vec.load('w2vf2model')

word_dic={}
with open(glove_path, encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split(' ')
        word_dic[line[0]] = {}
        vector = ''

        for i in range(1, len(line)):
            vector += ' ' + line[i]
        word_dic[line[0]] = vector

##################
# 生成新建词典库 #
################
dic = []
count = []
count_have=0
count_else=0
count_w2v=0
with open(read_txt, encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split(' ')
        for word in line:
            # print(word)
            if word not in dic:
                dic.append(word)
                if word in word_dic:
                    count_have+=1
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(word + word_dic[word] + '\n')
                elif word in model:
                    count_w2v+=1
                    tmp_vec=''
                    for i in model.wv[word]:
                        tmp_vec+=' '+str(i)
                    #print(tmp_vec)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(word +tmp_vec+ '\n')
                else:
                    count_else+=1
                    #print(word)
                    # if count_else==10:break

                    import numpy as np

                    tmp = np.random.random(embedding)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(word)
                        for i in tmp:
                            fin.write(' ' + str(i))
                        fin.write('\n')
print(len(dic))
print(count_have,count_w2v,count_else)
print(dic)

###############################
# 生成新建字典库，包括特征编码 #
##############################
with open(read_txt, encoding='utf-8') as f:
    for line in f.readlines():
        line = line.split('\n')[0].split(' ')
        for word in line:
            # print(word)
            if word not in dic:
                dic.append(word)
                words=word.split('/')
                if words[0] in word_dic:
                    count_have+=1
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(words[0] + word_dic[words[0]])
                elif words[0] in model:
                    print(words[0])
                    count_w2v+=1
                    tmp_vec=''
                    #print(model.wv[words[0]])
                    for i in model.wv[words[0]]:
                        tmp_vec+=' '+str(i)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(words[0] +tmp_vec)
                else:
                    count_else+=1
                    import numpy as np

                    tmp = np.random.random(embedding)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        fin.write(words[0])
                        for i in tmp:
                            fin.write(' ' + str(i))

                tmp_vec = ''
                if words[1] not in modelf1:
                    tmp=np.random.random(embedding)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        for i in tmp:
                            fin.write(' ' + str(i))
                for i in modelf1.wv[words[1]]:
                    tmp_vec += ' ' + str(i)
                # print(tmp_vec)
                with open(result_path, 'a', encoding='utf-8') as fin:
                    fin.write(tmp_vec)

                tmp_vec = ''
                if words[2] not in modelf2:
                    tmp = np.random.random(embedding)
                    with open(result_path, 'a', encoding='utf-8') as fin:
                        for i in tmp:
                            fin.write(' ' + str(i))
                for i in modelf2.wv[words[2]]:
                    tmp_vec += ' ' + str(i)
                # print(tmp_vec)
                with open(result_path, 'a', encoding='utf-8') as fin:
                    fin.write(tmp_vec+'\n')
print(len(dic))
print(count_have,count_w2v,count_else)