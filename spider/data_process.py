import json
import os
import jieba
import matplotlib.pyplot as plt
import re


def split_sentence(sentence):
    re_sentence_sp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')
    s = sentence
    slist = []
    for i in re_sentence_sp.split(s):
        if re_sentence_sp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def sentence_split(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


def read_text_from_corpus(corpus_path):
    data_list = []   # 所有读取的文件
    data_name = []   # 所有读取的文件名
    for root, dirs, files in os.walk(corpus_path):
        for file in files:
            with open(root + '/' + file) as f:
                value_name = file.strip('.txt')
                data_name.append(value_name)
                file_data_list = f.readlines()
                file_data_list = [line.replace('\n', '') for line in file_data_list]
                for index, line in enumerate(file_data_list):
                    # if not (line.endswith('。') or line.endswith('\"') or line.endswith('...') or line.endswith('。。。')
                    #         or line.endswith('？') or line.endswith('！') or line.endswith('?') or line.endswith('!')):
                    if line.endswith('html'):
                        file_data_list[index] = line + '。'
                file_data_str = ''.join(file_data_list)
                file_data_split = split_sentence(file_data_str)
                exec('{} = file_data_split '.format(value_name))
                exec('data_list.append({})'.format(value_name))
    return data_name, data_list


def sentence_length_summary():
    f = open('dataExtracted.txt', 'r', encoding='UTF-8')
    f_txt = f.readlines()
    f.close()
    data_name, data_list = read_text_from_corpus('corpus_txt/')
    max_length = 0
    length = []
    for data in data_list:
        for line in data:
            buf = jieba.lcut(line)
            if max_length < len(buf):
                max_length = len(buf)
            length.append(len(buf))
    length_num = [0] * max_length
    for item in length:
        length_num[item-1] = length_num[item-1] + 1
    sum = [0] * max_length
    for index, item in enumerate(length_num):
        if index == 0:
            sum[index] = length_num[index]
        else:
            sum[index] = sum[index-1] + length_num[index]
    print(sum)
    plt.bar(range(0, len(length_num)), [sum[i] for i in range(0, len(length_num))], label='all_sentence')
    plt.legend()
    plt.xlabel('length')
    plt.ylabel('num')
    plt.savefig('所有新闻句子长度分布直方图.png')


def json_to_txt():
    for i in range(1, 290):
        try:
            f = open('corpus/News'+str(i)+'.json', 'r')
            data = json.load(f)
            f.close()
        except:
            continue
        with open('corpus_txt/News'+str(i)+'.txt', 'w') as f:
            for article in data['articles']:
                f.write(article['article_content'].replace('\n', '') + '\n')


# 为brat的ann标注标记上对应行,同时修改为行的对应标记
# 返回标记好的ann数组，每一项为一个列表，表示一个标注的所有信息以及对应txt中行号
def labeled_data_process(txt, ann):
    f = open(txt, 'r', encoding='UTF-8')
    f_txt = f.readlines()
    f.close()
    f = open(ann, 'r', encoding='UTF-8')
    f_ann = f.readlines()
    f.close()
    start = 0
    end = 0
    last_length = 0
    # 将ann中所有相对文档的位置，转换为相对句子的位置，且匹配到对应句子
    for index1, sentence in enumerate(f_txt):
        start = start + last_length
        end = end + len(sentence) + 1
        last_length = len(sentence) + 1
        for index, label in enumerate(f_ann):
            label = label.strip('\n').replace('\t', ' ')
            buf = label.split(' ')
            if buf[0][0] == 'T':
                if int(buf[2]) >= start and int(buf[3]) <= end:
                    buf[2] = str(int(buf[2]) - start)
                    buf[3] = str(int(buf[3]) - start)
                    buf.append(str(index1))
            f_ann[index] = ' '.join(buf)
    return f_txt, f_ann


if __name__ == '__main__':
    # json_to_txt()
    # f_txt, f_ann = labeled_data_process('dataExtracted.txt', 'dataExtracted.ann')
    # for item in f_ann:
    #     print(item)
    sentence_length_summary()
