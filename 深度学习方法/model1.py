###################
# 模型搭建主要地区 #
###################
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from scipy.stats import pearsonr
max_len=600
vocal_size=7900#15645
embedding_size=300
class_num=1
embeded_txt='vec_new3.txt'
epoch=20
batch_size=16
feature_len=18
'''
如果需要加入文本句法分析等的向量标注，则需要扩充embedding_size,即可
'''

def read_train_file(train_path,tokenizer):
    '''
    :param train_path: 训练文件路径
    :param tokenizer: 由keras tokenizer 处理过的字典
    :return: 已处理的训练数据
    '''
    train_txt=[]
    train_label1=[]
    train_label2=[]
    train_label3=[]
    with open(train_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            train_txt.append((line[1]))
            train_label1.append(line[2])
            train_label2.append(line[3])
            train_label3.append(line[4])

    train_txt=tokenizer.texts_to_sequences(train_txt)
    train_txt=pad_sequences(train_txt,maxlen=max_len)
    train_label1=np.array(train_label1)
    train_label2 = np.array(train_label2)
    train_label3 = np.array(train_label3)
    print(train_txt.shape)
    return train_txt,train_label1,train_label2,train_label3

def read_test_file(test_path,tokenizer):
    '''
    :param test_path:测试文件的路径
    :param tokenizer:由keras tokenizer 处理过的字典
    :return: 已处理的测试数据
    '''
    test_x=[]
    with open(test_path,encoding='utf-8') as f:
        for line in f.readlines():
            line=line.split('\n')[0].split('\t')
            test_x.append(line[1])
    test_x=tokenizer.texts_to_sequences(test_x)
    test_x = pad_sequences(test_x, maxlen=max_len)
    return test_x


def create_embedding_matrix(vec_path):
    '''
    :param vec_path: 自己建的词向量路径
    :return:词向量矩阵及生成的字典
    '''
    word_vec = {}
    word_dic = []
    with open(vec_path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.split('\n')[0].split(' ')
            word_vec[line[0]] = {}
            vec = []
            for i in range(1, embedding_size+1):
                vec.append(float(line[i]))
            word_vec[line[0]] = vec
            word_dic.append(line[0])
    embedding_matrix = np.zeros((vocal_size + 1, embedding_size))

    tokenizer = Tokenizer(num_words=vocal_size, lower=False)
    tokenizer.fit_on_texts(word_dic)
    word_index = tokenizer.word_index
    for word, i in word_index.items():
        embedding_matrix[i] = word_vec[word]
    return embedding_matrix, tokenizer

def build_model(embeded_matrix):
    '''
    主要模型搭建，采用3层cnn加LSTM构成，另外可支持以有特征输入,特征的网络就简单的Dense。
    最后两个模型融合同时输出
    :param embeded_matrix: 词向量矩阵
    :return: 模型框架
    '''
    cnn_filter=1024# cnn 输出大小
    cnn_kernel=3   # cnn 核
    lstm_unit=512 # LSTM 输出维度

    from keras.layers import Conv1D,MaxPool1D,LSTM,Dense,concatenate
    from keras.layers import Input,Embedding,Dropout
    from keras.models import Model
    from keras import initializers

    ##############
    # w2v input #
    ############
    sequence_input=Input(shape=(max_len,),dtype='int32')
    embedding_layer=Embedding(vocal_size+1,embedding_size,
                              weights=[embeded_matrix],
                              input_length=max_len)

    embed_sequence=embedding_layer(sequence_input)
    embed_sequence=Dropout(0.3)(embed_sequence)
    x=Conv1D(filters=cnn_filter,kernel_size=2,activation='relu',kernel_initializer=initializers.glorot_uniform(seed=None))(embed_sequence)
    x=MaxPool1D(cnn_kernel)(x)
    x = Conv1D(filters=cnn_filter, kernel_size=3, activation='relu',kernel_initializer=initializers.glorot_uniform(seed=None))(x)
    x = MaxPool1D(cnn_kernel)(x)
    x = Conv1D(filters=cnn_filter, kernel_size=5, activation='relu',kernel_initializer=initializers.glorot_uniform(seed=None))(x)
    x = MaxPool1D(cnn_kernel)(x)
    x = LSTM(lstm_unit)(x)
    x=Dense(512,activation='relu' ,kernel_initializer=initializers.glorot_uniform(seed=None))(x)
    x=Dropout(0.2)(x)


    ##################
    # feature input #
    #################
    feature_input=Input(shape=(feature_len,))
    x2=Dense(512,activation='relu',kernel_initializer=initializers.glorot_uniform())(feature_input)
    x2=Dropout(0.5)(x2)
    x2=Dense(256,activation='relu',kernel_initializer=initializers.glorot_uniform())(x2)
    x2=Dropout(0.5)(x2)
    x2=Dense(128,activation='relu',kernel_initializer=initializers.glorot_uniform())(x2)
    x2=Dropout(0.5)(x2)
    merge=concatenate([x,x2])
    merge=Dense(1024,activation='relu')(merge)
    merge=Dropout(0.5)(merge)
    merge=Dense(512,activation='relu')(merge)
    merge=Dropout(0.5)(merge)
    preds=Dense(1,activation='relu')(merge)

    ##########
    # model #
    ########
    model=Model(inputs=[sequence_input,feature_input],outputs=preds)
    model.compile(loss='mse',optimizer='nadam',metrics=['mae'])
    return model



def create_model1(model1):
    '''
    与下一个函数基本一致，使用两组模型的原因是防止在同一个模型上反复训练，用两个模型模拟两组打分
    :param model1: 由buildmodel传入模型
    '''
    from keras.callbacks import TensorBoard
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    print(model1.summary())
    print(train_x2.shape)
    model1.fit([train_x,train_x2],train_label1,epochs=epoch,batch_size=batch_size,validation_data=[[vaild_x,vaild_x2],vaild_label1],
               callbacks=[TensorBoard(log_dir='./tmp/log1')])
    model1.save(model_path1)
    print('model1 finish')
def create_model2(model2):
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model2.fit([train_x,train_x2],train_label2, epochs=epoch, batch_size=batch_size, validation_data=[[vaild_x,vaild_x2],vaild_label2],callbacks=[early_stopping])
    model2.save(model_path2)
    print('model2 finish')

def result_file(save_path,test_path,preds1,preds2):
    '''
    对模型结果进行评估，采用peason的评估方式
    :param save_path: 缓存一个文件，观察数值
    :param test_path: 测试文件地址
    :param preds1: 第一组老师的预测值
    :param preds2: 第二组老师的预测值
    '''
    id_list=[]
    answer=[]
    a2=[]
    f_a=[]
    right=[]
    txt=[]
    count=0
    with open(test_path,encoding='utf-8') as f:
        for line in f.readlines():
            count+=1
            if count<=train_num:continue
            # print(line.split('\t')[0])
            id_list.append(line.split('\t')[0])
            txt.append(line.split('\t')[1])
            answer.append(float(line.split('\t')[2]))
            a2.append(float(line.split('\t')[3]))
            f_a.append(float(line.split('\t')[2])+float(line.split('\t')[3]))
            right.append(float(line.split('\t')[4].split('\n')[0]))
    out = []
    out1 = []
    out2 = []
    # out3=[]
    for i in range(1200-train_num):
        out1.append(preds1[i][0])
        out2.append(preds2[i][0])
        out.append(preds2[i][0] + preds1[i][0])
    print(len(out),len(right))
    with open(save_path,'w',encoding='utf-8') as f:
        for i in range(1200-train_num):

            f.write(str(id_list[i])+'\t'+str(answer[i])+'\t'+str(preds1[i][0])+'\t'+str(a2[i])+'\t'+str(preds2[i][0])+'\t'+str(preds1[i][0]+preds2[i][0])+'\n')
    print(pearsonr(out1,answer)[0])
    print(pearsonr(out2,a2)[0])
    print(pearsonr(out,right)[0])


def submission_file(test_path,preds1,preds2):
    '''
    生成提交文件
    '''
    id_list=[]
    with open(test_path,encoding='utf-8') as f:
        for line in f.readlines():
            id_list.append(line.split('\t')[0])

    with open('submission_sample.txt','w',encoding='utf-8') as f:
        for i in range(len(id_list)):

            f.write(str(id_list[i])+','+str(preds1[i][0]+preds2[i][0])+'\n')


##################################
# 一些杂七杂八的模块，可以进行调整 #
##################################
train_num=1100 #训练1100组数据

model_path1='model213_save'#'model11_save'
model_path2='model223_save'#'model12_save'
model_path3='model13_save'
train_path='ctrain.txt'
test_path='ctest.txt'
vec_path='vec_new3.txt'
save_path='tmp.txt'

##################################
# 读取特征文件，并对特征进行归一化 #
##################################
import numpy as np
import pickle
from sklearn import preprocessing
with open('train.pickle','rb') as f:
    test=pickle.load(f)
train_x2=[]
for i in range(len(test)):
    train_x2.append(preprocessing.scale(test[i][0]))

train_x2=np.array(train_x2)

with open('test.pickle','rb') as f:
    test=pickle.load(f)
test_x2=[]
for i in range(len(test)):
    test_x2.append(preprocessing.scale(test[i][0]))
test_x2=np.array(test_x2)

print(test_x2)


matrix,tokenizer=create_embedding_matrix(vec_path)

model1=load_model(model_path1)
model2=load_model(model_path2)

model1=build_model(matrix)
from keras.utils import plot_model
plot_model(model1,to_file='model.png')
#model2=build_model(matrix)
# print(model1.summary())
# model3=build_model(matrix)

train_x,train_label1,train_label2,train_label3=read_train_file(train_path,tokenizer)

#####################
# 训练集，验证集划分 #
#####################

train=train_x
vaild_x=train_x[train_num:]
vaild_x2=train_x2[train_num:]
vaild_label1=train_label1[train_num:]
vaild_label2=train_label2[train_num:]
vaild_label3=train_label3[train_num:]

train_x=train_x[:train_num]
train_x2=train_x2[:train_num]
train_label1=train_label1[:train_num]
train_label2=train_label2[:train_num]
train_label3=train_label3[:train_num]



create_model1(model1)
create_model2(model2)

################
# 预测文件部分 #
###############

preds1=model1.predict([vaild_x,vaild_x2],batch_size=batch_size)

preds2=model2.predict([vaild_x,vaild_x2],batch_size=batch_size)
result_file(save_path,'ctrain.txt',preds1,preds2)
test=read_test_file(test_path,tokenizer)
preds1=model1.predict([test,test_x2],batch_size)
preds2=model2.predict([test,test_x2],batch_size)
submission_file(test_path,preds1,preds2)
