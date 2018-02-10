
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

string="I don't know"
new_string=clean_str(string)
print(new_string)
# print(tokenize(new_string))


result_path='all_data.txt'
train_path='train.txt'
test_path='test.txt'
c_train='ctrain.txt'
c_test='xtest.txt'

###########################
# 组成新的已清理的训练文件 #
##########################
with open(result_path,'w',encoding='utf-8') as fin:
    with open(c_train,'w',encoding='utf-8') as fintrain:
        with open(train_path,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\t')
                new_string=clean_str(line[1])
                fin.write(new_string+'\n')
                fintrain.write(line[0]+'\t'+new_string+'\t'+line[2]+'\t'+line[3]+'\t'+line[4])
    with open(c_test,'w',encoding='utf-8') as fintest:
        with open(test_path,encoding='utf-8') as f:
            for line in f.readlines():
                line=line.split('\n')[0].split('\t')
                new_string=clean_str(line[1])
                fin.write(new_string+'\n')
                fintest.write(line[0]+'\t'+new_string+'\n')


###################################
# 对于文本特征构建组成新的训练文件 #
#################################
from textblob import TextBlob
with open('f1.txt','a',encoding='utf-8') as f1in:
    with open('f2.txt', 'a', encoding='utf-8') as f2in:
        with open('f3.txt', 'a', encoding='utf-8') as f3in:
            with open('ctrain.txt',encoding='utf-8') as f:
                count=0
                for line in f.readlines():
                    count+=1
                    text=line.split('\n')[0].split('\t')[1]
                    tag=TextBlob(text)
                    # print(tag.tags)
                    print(tag.parse().split(' '))
                    f=tag.parse().split(' ')
                    for i in f:
                        i=i.split('/')
                        f1in.write(i[1]+' ')
                        f2in.write(i[2]+' ')
                        f3in.write(i[3]+' ')
                f1in.write('\n')
                f2in.write('\n')
                f3in.write('\n')

                    # print(line)
                    # f count==10:break
from textblob import TextBlob
with open('ftmp_test.txt','w',encoding='utf-8') as fin:
    with open('tmp_test',encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            text=line[1]
            tag=TextBlob(text)
            line[1]=tag.parse().replace('\n',' ')
            for i in line:
                fin.write(i+'\t')
            fin.write('\n')