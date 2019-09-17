# -*- coding: UTF-8 -*-
import time
import re
import operator
import os
import csv
from bert_base.client import BertClient
from bert_base.bert import tokenization
from bert_base.train.train_helper import get_args_parser
args = get_args_parser()
bert_dir = r'NER_model/chinese_L-12_H-768_A-12'
tokenizer = tokenization.FullTokenizer(
    vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)
bc = BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER')


# 中文分句
def sentence_split(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split('\n')
    res = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if len(sentence) != 0:
            res.append(sentence)
    return res


# 统计文本中所有实体出现的频数
def get_all_entity_list(news_path, res_path):
    entity_time = {}
    with open(news_path, 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        # 对于文件中的所有事件
        for index, item in enumerate(f_csv):
            content = item[3]
            # 进行分句
            sentences = sentence_split(content)
            # 对所有句子进行命名实体识别
            try:
                all_entities = get_ner_list(sentences)
            except Exception as ec:
                with open(r'LawNews/NER_log.txt', 'a', encoding='utf-8') as log:
                    log.write(str(index) + '\t' + item[0] + '\n')
                    log.write(str(ec) + '\n')

            # 统计识别结果
            for (entities, sentence) in zip(all_entities, sentences):
                # print(sentence)
                for entity in entities['entities']:
                    # print(entity['word'], entity['start'], entity['end'], entity['type'])
                    word = entity['word']
                    if word not in entity_time:
                        entity_time[word] = 1
                    else:
                        entity_time[word] = entity_time[word] + 1

            # 每100条新闻存储一次结果，防止丢失
            if (index + 1) % 100 == 0:
                print('--------------------当前已处理', index+1, '条新闻--------------------')
                with open(res_path, 'w', newline='', encoding='utf-8') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(['命名实体', '出现次数'])
                    print('总实体个数：', entity_time)
                    for key in entity_time:
                        f_csv.writerow([key, entity_time[key]])
    # entity_time = sorted(entity_time.items(), key=operator.itemgetter(1), reverse=True)


# 对get_all_entity_list函数中出现的错误情况进行处理
def error_process(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        indexes = []
        for line in all_lines:
            line = line.strip('\n')
            index = line.split('\t')[0]
            if index.isdigit():
                indexes.append(int(index))

    with open('LawNews/all_law_news_content_list.csv', 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        entity_time = {}
        for index, item in enumerate(f_csv):
            if index in indexes:
                sentences = sentence_split(item[3])
                if len(sentences) == 0:
                    print(index, "新闻分句结果为空")
                    continue
                print(sentences)
                all_entities = get_ner_list(sentences)
                print(all_entities)
                # 统计识别结果
                for entities in all_entities:
                    for entity in entities['entities']:
                        word = entity['word']
                        if word not in entity_time:
                            entity_time[word] = 1
                        else:
                            entity_time[word] = entity_time[word] + 1

    with open('LawNews/entities_list_error_process.csv', 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['命名实体', '出现次数'])
        print('总实体个数：', len(entity_time))
        for key in entity_time:
            f_csv.writerow([key, entity_time[key]])


# 给定一个句子的集合，返回这些句子的NER的结果，返回结果为json格式
def get_ner_list(sentences):
    rst = bc.encode(sentences)
    res = NER_Result()
    entities = []
    # print('rst:', rst)
    for (one_str, one_rst) in zip(sentences, rst):
        ners = res.result_to_json(tokenizer.tokenize(one_str), one_rst)
        entities.append(ners)
    return entities


# 根据句子以及NER的结果编码，返回json格式的NER结果
class NER_Result(object):
    def __init__(self):
        self.person = []
        self.loc = []
        self.org = []
        self.others = []

    def get_result(self, tokens, tags, config=None):
        # 先获取标注结果
        self.result_to_json(tokens, tags)
        return self.person, self.loc, self.org

    def result_to_json(self, string, tags):
        """
        将模型标注序列和输入序列结合 转化为结果
        :param string: 输入序列
        :param tags: 标注结果
        :return:
        """
        item = {"entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        last_tag = ''

        for char, tag in zip(string, tags):
            if tag[0] == "S":
                self.append(char, idx, idx+1, tag[2:])
                item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
            elif tag[0] == "B":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "O":
                if entity_name != '':
                    self.append(entity_name, entity_start, idx, last_tag[2:])
                    item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                    entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
            last_tag = tag
        if entity_name != '':
            self.append(entity_name, entity_start, idx, last_tag[2:])
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
        return item

    def append(self, word, start, end, tag):
        if tag == 'LOC':
            self.loc.append(Pair(word, start, end, 'LOC'))
        elif tag == 'PER':
            self.person.append(Pair(word, start, end, 'PER'))
        elif tag == 'ORG':
            self.org.append(Pair(word, start, end, 'ORG'))
        else:
            self.others.append(Pair(word, start, end, tag))


# 用于表示NER结果的数据类型
class Pair(object):
    def __init__(self, word, start, end, type, merge=False):
        self.__word = word
        self.__start = start
        self.__end = end
        self.__merge = merge
        self.__types = type

    @property
    def start(self):
        return self.__start
    @property
    def end(self):
        return self.__end
    @property
    def merge(self):
        return self.__merge
    @property
    def word(self):
        return self.__word

    @property
    def types(self):
        return self.__types
    @word.setter
    def word(self, word):
        self.__word = word
    @start.setter
    def start(self, start):
        self.__start = start
    @end.setter
    def end(self, end):
        self.__end = end
    @merge.setter
    def merge(self, merge):
        self.__merge = merge

    @types.setter
    def types(self, type):
        self.__types = type

    def __str__(self) -> str:
        line = []
        line.append('entity:{}'.format(self.__word))
        line.append('start:{}'.format(self.__start))
        line.append('end:{}'.format(self.__end))
        line.append('merge:{}'.format(self.__merge))
        line.append('types:{}'.format(self.__types))
        return '\t'.join(line)


# 抽取并记录所有新闻中的命名实体
def get_entity_in_every_news(news_path, res_path):
    f2 = open(res_path, 'w', newline='', encoding='utf-8')
    f2_csv = csv.writer(f2)
    f2_csv.writerow(['新闻序号', '实体字典'])
    with open(news_path, 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        # 对于文件中的所有事件
        for index, item in enumerate(f_csv):
            content = item[3]
            # 进行分句
            sentences = sentence_split(content)
            if len(sentences) == 0:
                print(index, "新闻分句结果为空")
                continue
            # 对所有句子进行命名实体识别
            all_entities = get_ner_list(sentences)
            entity_time = {}
            for entities in all_entities:
                for entity in entities['entities']:
                    word = entity['word']
                    if word not in entity_time:
                        entity_time[word] = 1
                    else:
                        entity_time[word] = entity_time[word] + 1
            print(index, str(entity_time))
            f2_csv.writerow([index, str(entity_time)])
    f2.close()


# 合并两个entities list
def entities_list_combine():
    f1 = open('LawNews/entities_list.csv', 'r', newline='', encoding='utf-8')
    f2 = open('LawNews/entities_list_error_process.csv', 'r', newline='', encoding='utf-8')
    f1_csv = csv.reader(f1)
    f2_csv = csv.reader(f2)
    entities_list = {}
    for index, item in enumerate(f1_csv):
        if index > 0:
            entities_list[item[0]] = int(item[1])
    for index, item in enumerate(f2_csv):
        if index > 0:
            if item[0] not in entities_list:
                entities_list[item[0]] = int(item[1])
            else:
                entities_list[item[0]] = entities_list[item[0]] + int(item[1])
    with open('LawNews/all_law_news_entities_list.csv', 'w', newline='', encoding='utf-8') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(['命名实体', '出现次数'])
        for key in entities_list:
            f_csv.writerow([key, entities_list[key]])
    f1.close()
    f2.close()



if __name__ == '__main__':
    # error_process('LawNews/NER_log.txt')
    get_entity_in_every_news('LawNews/all_law_news_content_list.csv', 'LawNews/every_news_entities_list.csv')
    # entities_list_combine()