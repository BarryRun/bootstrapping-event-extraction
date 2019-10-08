# -*- coding: UTF-8 -*-
import re
import os
import random
import csv
from bert_base.client import BertClient
from bert_base.bert import tokenization
from bert_base.train.train_helper import get_args_parser


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
    args = get_args_parser()
    bert_dir = r'NER_model/chinese_L-12_H-768_A-12'
    tokenizer = tokenization.FullTokenizer(
        vocab_file=os.path.join(bert_dir, 'vocab.txt'), do_lower_case=args.do_lower_case)
    bc = BertClient(show_server_config=False, check_version=False, check_length=False, mode='NER')
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
    f2 = open(res_path, 'a', newline='', encoding='utf-8')
    f2_csv = csv.writer(f2)
    # f2_csv.writerow(['新闻序号', '实体字典'])
    with open(news_path, 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        # 对于文件中的所有事件
        for index, item in enumerate(f_csv):
            if index <= 66513:
                continue
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


# 计算所有文本之间的聚类评分s
class ClusterResult(object):
    def __init__(self, all_law_news_entities_path, all_law_news_content_list, entities_path, res_path):
        self.all_law_news_entities_path = all_law_news_entities_path
        self.all_law_news_content_list = all_law_news_content_list
        self.entities_path = entities_path
        self.res_path = res_path
        self.all_entities_dic = {}
        self.every_entities_dict = {}
        self.news_list = []

    def loading_data(self):
        # 加载整体的命名实体字典
        with open(self.all_law_news_entities_path, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f)
            for index, item in enumerate(f_csv):
                if index > 0:
                    self.all_entities_dic[item[0]] = int(item[1])

        # 加载每个新闻的命名实体列表
        with open(self.entities_path, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f)
            for index, item in enumerate(f_csv):
                if index > 0:
                    self.every_entities_dict[int(item[0])] = eval(item[1])

        # 加载所有新闻（主要是为了统计同一天的新闻）
        with open(self.all_law_news_content_list, 'r', newline='', encoding='utf-8') as f:
            f_csv = csv.reader(f)
            for item in f_csv:
                self.news_list.append([item[0], item[1], item[2], item[3]])

    def calculate_score(self):
        self.loading_data()
        with open(self.res_path, 'w', newline='', encoding='utf-8') as f:
            f_csv = csv.writer(f)
            for index1 in range(len(self.news_list)):
                if index1 not in self.every_entities_dict:
                    continue
                # date = self.news_list[index1][2]
                for index2 in range(index1+1, len(self.news_list)):
                    # if date == self.news_list[index2][2] and index2 in self.every_entities_dict:
                    # 不做日期的限制
                    if index2 in self.every_entities_dict:
                        s = self.calculate_score_between_news(index1, index2)
                        if s != 0:
                            f_csv.writerow([index1, index2, s])
                print(index1, "over")

    # 计算两个文档之间的聚类评分
    def calculate_score_between_news(self, index1, index2):
        entities1 = self.every_entities_dict[index1]
        entities2 = self.every_entities_dict[index2]
        same_entities_dic = []
        cross_time = 0
        for key1 in entities1:
            for key2 in entities2:
                if key1 == key2:
                    cross_time += entities2[key2] + entities1[key1]
                    same_entities_dic.append(key1)
        if cross_time == 0:
            return 0
        all_time = 0
        for entity in same_entities_dic:
            all_time += self.all_entities_dic[entity]

        return cross_time/all_time


# 为所有文档分类
def document_cluster(score_path, document_cluster_path):
    clusters = []
    with open(score_path, 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for item in f_csv:
            # 界定一个阈值
            if float(item[2]) > 0.5:
                existing = False
                for cluster in clusters:
                    if item[0] in cluster:
                        if item[1] not in cluster:
                            cluster.append(item[1])
                        existing = True
                        break
                    else:
                        if item[1] in cluster:
                            cluster.append(item[0])
                            existing = True
                            break
                if not existing:
                    clusters.append([item[0], item[1]])

    with open(document_cluster_path, 'w', encoding='utf-8') as f:
        for item in clusters:
            f.write(str(item) + '\n')


# 对事件对齐的结果进行一定的分析
def result_analysis():
    news_list = []
    # 加载所有新闻（主要是为了统计同一天的新闻）
    with open('LawNews/all_law_news_content_list.csv', 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for item in f_csv:
            news_list.append([item[0], item[1], item[2], item[3]])

    every_entities_dict = {}
    # 加载每个新闻的命名实体列表
    with open('LawNews/every_news_entities_list.csv', 'r', newline='', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for index, item in enumerate(f_csv):
            if index > 0:
                every_entities_dict[int(item[0])] = eval(item[1])

    random_numbers = []
    for i in range(20):
        random_numbers.append(random.randint(0, 97))
    with open('LawNews/news_cluster_res_threshold_0.1.txt') as f:
        for index, line in enumerate(f.readlines()):
            if index not in random_numbers:
                continue
            print('========Cluster' + str(index) + '========')
            print(line.strip())
            tmp = line[1:-2].split(', ')
            # if len(tmp) > 10:
            #     continue

            for item in tmp:
                news_index = int(item[1:-1])
                print(news_index, news_list[news_index][0], news_list[news_index][2], every_entities_dict[news_index])
                print(news_list[news_index][3])


if __name__ == '__main__':
    # error_process('LawNews/NER_log.txt')
    # get_entity_in_every_news('LawNews/all_law_news_content_list.csv', 'LawNews/every_news_entities_list.csv')
    # entities_list_combine()
    # res_class = ClusterResult('LawNews/all_law_news_entities_list.csv',
    #                           'LawNews/all_law_news_content_list.csv',
    #                           'LawNews/every_news_entities_list.csv',
    #                           'LawNews/cluster_score_res_without_date_limit.csv')
    # res_class.calculate_score()
    # document_cluster('LawNews/cluster_score_res_with_date_limit.csv',
    #                  'LawNews/news_cluster_res_threshold_0.5.txt')
    result_analysis()
