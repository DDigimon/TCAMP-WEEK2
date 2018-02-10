# TCAMP-WEEK2-自动评分系统
好未来第二周
DL-naive
第二周内容
自动评分
这次数据集提供围绕特定主题的英语文章文本数据，可用于针对学生所写的英文作文进行机器自动评分。
由于对于应试教育作文打分有一定规则，因而可以通过机器来学习这样的规则。每个老师对于作文评分
在1至6之间，一篇作文由两位老师打分，之后相加得到作文总分。最后提交为这篇作文总分

项目提供txt文件，每列由tab隔开。每行为一个样例
训练集共1200个样本，5个字段，分别为编号，文本，评分1，评分2，作文总分
示例：
1 "Dear Local Newspaper, I believe the ……” 4 4 8
2 "Dear @CAPS1 @CAPS2, I have heard ……” 6 4 10
测试集共550样本，2个字段，分别为编号，文本。
示例：
1 "Dear Local Newspaper, I believe the ……”
2 "Dear @CAPS1 @CAPS2, I have heard ……”
