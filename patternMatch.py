import os
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


def regular_expression_match(re_sentence, file):
    regress = re.compile(re_sentence)
    res = []
    for sentence in file:
        buf = regress.findall(sentence)
        res.extend(buf)
    return res


if __name__ == '__main__':
    newName, newsList = read_text_from_corpus('spider/corpus_txt')
    # f = open('dataExtracted.txt', 'w')
    for (name, news) in zip(newName, newsList):
        # buf = regular_expression_match('战胜', news)
        # re_res.extend(buf)
        for new in news:
            if "战胜" in new:
                # f.write(name + ' ' + new + '\n')
                # f.write(new + '\n')
                print(name, new)
    # f.close()
