import codecs
import random
import copy
import jieba
import gc
import numpy as np
from pyhanlp import HanLP
# from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import tensorflow as tf
# from bert import tokenization, modeling
# from bert_embedding import BertEmbedding
import os
from spider import data_process
jieba.load_userdict('userDict.txt')


# 语料库所有数据的读取
def corpus_sentence():
    embeddings_index, embeddings_length = get_chinese_embedding()
    file_name, file_data = data_process.read_text_from_corpus('spider/corpus_txt/')
    all_sentence_data = {}
    test_num = 0
    for (name, data) in zip(file_name, file_data):
        if test_num > 1:
            break
        all_sentence_data[name] = []
        for sentence_index, sentence in enumerate(data):
            before_length = 0
            words = jieba.lcut(sentence)
            for index, word in enumerate(words):
                one_data = {}
                one_data['sentence_index'] = sentence_index
                one_data['word'] = word
                one_sentence_feature = sentence_feature_input(sentence, embeddings_index, True,
                                                              word, before_length)
                if one_sentence_feature is not None:
                    one_data['sentence_vector'] = one_sentence_feature
                    one_data['lexical_vector'] = lexical_level_feature(sentence, embeddings_index, embeddings_length,
                                                                       word, before_length)
                    # all_sentence_data[name].append(one_sentence_feature)
                before_length = before_length + len(word)
                all_sentence_data[name].append(copy.deepcopy(one_data))
                one_data.clear()
        test_num = test_num + 1
    # print(np.asarray(all_sentence_data['News1'][0]['vector']).shape)


# 触发词的训练数据
def trigger_data(positive_data, negative_data, first_iteration=False):
    embeddings_index, embeddings_length = get_chinese_embedding()
    # positive_sentence_before = []
    # positive_sentence_after = []
    # negative_sentence_before = []
    # negative_sentence_after = []
    positive_lexical_level = []
    negative_lexical_level = []
    positive_sentence_level = []
    negative_sentence_level = []
    # f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    # # 标注数据中获取的训练数据
    # for ann in f_ann:
    #     buf = ann.split(' ')
    #     print(buf)
    #     # 关于触发词的训练数据
    #     if buf[1] == 'Game':
    #         line = int(buf[-1])
    #         print(f_txt[line])
    #         one_sentence_before, one_sentence_after = sentence_feature_input(f_txt[line], embeddings_index, True,
    #                                                                          buf[-2], int(buf[2]))
    #         if one_sentence_before is not None and one_sentence_after is not None:
    #             positive_sentence_before.append(copy.deepcopy(one_sentence_before))
    #             positive_sentence_after.append(copy.deepcopy(one_sentence_after))
    #             positive_lexical_level.append(lexical_level_feature(f_txt[line], embeddings_index,
    #                                           embeddings_length, buf[-2], int(buf[2])))
    #     elif buf[1] == 'N-Game':
    #         line = int(buf[-1])
    #         print(f_txt[line])
    #         one_sentence_before, one_sentence_after = sentence_feature_input(f_txt[line], embeddings_index, True,
    #                                                                          buf[-2], int(buf[2]))
    #         if one_sentence_before is not None and one_sentence_after is not None:
    #             negative_sentence_before.append(copy.deepcopy(one_sentence_before))
    #             negative_sentence_after.append(copy.deepcopy(one_sentence_after))
    #             negative_lexical_level.append(lexical_level_feature(f_txt[line], embeddings_index,
    #                                           embeddings_length, buf[-2], int(buf[2])))

    # 通过聚类获取的训练数据
    if first_iteration is False:
        for cluster_data in positive_data:
            one_sentence_feature = sentence_feature_input(cluster_data['sentence'],
                                                          embeddings_index, True,
                                                          cluster_data['trigger'],
                                                          cluster_data['trigger_site'])
            one_lexical_feature = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                        embeddings_length, cluster_data['trigger'],
                                                        cluster_data['trigger_site'])
            if one_sentence_feature is not None and one_lexical_feature is not None:
                positive_sentence_level.append(copy.deepcopy(one_sentence_feature))
                # positive_sentence_before.append(copy.deepcopy(one_sentence_before))
                # positive_sentence_after.append(copy.deepcopy(one_sentence_after))
                positive_lexical_level.append(copy.deepcopy(one_lexical_feature))
        for cluster_data in negative_data:
            one_sentence_feature = sentence_feature_input(cluster_data['sentence'],
                                                                             embeddings_index, True,
                                                                             cluster_data['trigger'],
                                                                             cluster_data['trigger_site'])
            one_lexical_feature = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                        embeddings_length, cluster_data['trigger'],
                                                        cluster_data['trigger_site'])
            if one_sentence_feature is not None  and one_lexical_feature is not None:
                negative_sentence_level.append(copy.deepcopy(one_sentence_feature))
                # negative_sentence_before.append(copy.deepcopy(one_sentence_before))
                # negative_sentence_after.append(copy.deepcopy(one_sentence_after))
                negative_lexical_level.append(copy.deepcopy(one_lexical_feature))

    sentence_level = positive_sentence_level + negative_sentence_level
    # sentence_after = positive_sentence_after + negative_sentence_after
    lexical_level = positive_lexical_level + negative_lexical_level
    label = [1] * len(positive_sentence_level) + [0] * len(negative_sentence_level)
    data_length = len(sentence_level)

    # 对数据做shuffle
    index = [i for i in range(data_length)]
    random.shuffle(index)
    sentence_level = [sentence_level[i] for i in index]
    # sentence_before = [sentence_before[i] for i in index]
    # sentence_after = [sentence_after[i] for i in index]
    lexical_level = [lexical_level[i] for i in index]
    label = [label[i] for i in index]

    # 对shape进行一定的调整
    sentence_level = np.asarray(sentence_level, dtype='float32')
    sentence_level = np.reshape(sentence_level, (data_length, 110, 301, 1))
    # sentence_after = np.asarray(sentence_after, dtype='float32')
    # sentence_after = np.reshape(sentence_after, (data_length, 110, 301, 1))
    lexical_level = np.asarray(lexical_level, dtype='float32')
    label = np.asarray(label)
    return sentence_level, lexical_level, label


# 事件论元的训练数据
def argument_data(positive_data, negative_data, argument_type, first_iteration=False):
    embeddings_index, embeddings_length = get_chinese_embedding()
    # positive_sentence_before = []
    # positive_sentence_between = []
    # positive_sentence_after = []
    # negative_sentence_before = []
    # negative_sentence_between = []
    # negative_sentence_after = []
    positive_lexical_level = []
    negative_lexical_level = []
    positive_sentence_level = []
    negative_sentence_level = []
    # f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    # all_label_argument = {}
    # # 读取所有标注，并以字典的形式存储
    # for ann in f_ann:
    #     one_label = {}
    #     buf = ann.split()
    #     if buf[0][0] == 'T':
    #         # one_label['index'] = buf[0]
    #         one_label['type'] = buf[1]
    #         one_label['begin_site'] = buf[2]
    #         one_label['end_site'] = buf[3]
    #         one_label['content'] = buf[4]
    #         one_label['sentence_index'] = buf[5]
    #         sentence = f_txt[int(one_label['sentence_index'])].strip('\n')
    #         one_label['sentence_content'] = sentence
    #     elif buf[0][0] == 'E':
    #         # one_label['index'] = buf[0]
    #         for i in range(1, len(buf)):
    #             arguments = buf[i].split(':')
    #             one_label[arguments[0]] = arguments[1]
    #     all_label_argument[str(buf[0])] = copy.deepcopy(one_label)
    #
    # # 找出所有对应的trigger与argument
    # for key in all_label_argument.keys():
    #     label_data = all_label_argument[key]
    #     # 对于标注数据中的每个正例数据
    #     if key[0] == 'E' and len(label_data) == 5:
    #         one_argument = all_label_argument[label_data[argument_type]]
    #         one_trigger = all_label_argument[label_data['Game']]
    #         argument_site = int(one_argument['begin_site'])
    #         argument = one_argument['content']
    #         trigger_site = int(one_trigger['begin_site'])
    #         trigger = one_trigger['content']
    #         sentence = one_trigger['sentence_content']
    #         one_sentence_before, one_sentence_between, one_sentence_after = \
    #             sentence_feature_input(sentence, embeddings_index,  False, trigger, trigger_site,
    #                                    argument, argument_site)
    #         one_lexical_feature1 = lexical_level_feature(sentence, embeddings_index,
    #                                                      embeddings_length, trigger,
    #                                                      trigger_site)
    #         one_lexical_feature2 = lexical_level_feature(sentence, embeddings_index,
    #                                                      embeddings_length, argument,
    #                                                      argument_site)
    #         if one_sentence_before is not None and one_lexical_feature2 is not None and one_lexical_feature1 is not None:
    #             positive_sentence_before.append(copy.deepcopy(one_sentence_before))
    #             positive_sentence_between.append(copy.deepcopy(one_sentence_between))
    #             positive_sentence_after.append(copy.deepcopy(one_sentence_after))
    #             positive_lexical_level.append(one_lexical_feature1 + one_lexical_feature2)
    #
    # print("-------------标注数据获取完毕-------------")
    # 通过聚类获取的训练数据
    if argument_type == 'WinnerScore':
        argument_type = 'winner_score'
        argument_site = 'winner_score_site'
        word_nature = 'm'
    elif argument_type == 'LoserScore':
        argument_type = 'loser_score'
        argument_site = 'loser_score_site'
        word_nature = 'm'
    elif argument_type == 'Winner':
        argument_type = 'winner'
        argument_site = 'winner_site'
        word_nature = 'n'
    elif argument_type == 'Loser':
        argument_type = 'loser'
        argument_site = 'loser_site'
        word_nature = 'n'
    else:
        raise ValueError('argument type error')

    if first_iteration is False:
        for cluster_data in positive_data:
            if int(cluster_data[argument_site]) == -1:
                continue
            one_sentence_feature = sentence_feature_input(cluster_data['sentence'], embeddings_index, False, cluster_data['trigger'],
                                         int(cluster_data['trigger_site']), cluster_data[argument_type],
                                         int(cluster_data[argument_site]))
            one_lexical_feature1 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                         embeddings_length, cluster_data['trigger'],
                                                         int(cluster_data['trigger_site']))
            one_lexical_feature2 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                         embeddings_length, cluster_data[argument_type],
                                                         int(cluster_data[argument_site]))
            if one_sentence_feature is not None and one_lexical_feature2 is not None and one_lexical_feature1 is not None:
                positive_sentence_level.append(copy.deepcopy(one_sentence_feature))
                # positive_sentence_between.append(copy.deepcopy(one_sentence_between))
                # positive_sentence_after.append(copy.deepcopy(one_sentence_after))
                positive_lexical_level.append(one_lexical_feature1 + one_lexical_feature2)
            sentence_parse = HanLP.segment(cluster_data['sentence'])
            site = 0
            # print("-------------单个聚类结果正例数据获取完毕-------------")
            # 将句子中的其他实体作为负例训练数据
            for parsed_word in sentence_parse:
                if site == cluster_data['trigger_site']:
                    site = site + len(parsed_word.word)
                    continue
                # 随机选择是否将该实体作为负例（0.5的概率）
                if random.randint(0, 9) < 3:
                    site = site + len(parsed_word.word)
                    continue
                if str(parsed_word.nature) == word_nature:
                    one_sentence_feature \
                        = sentence_feature_input(cluster_data['sentence'], embeddings_index, False,
                                                 cluster_data['trigger'],
                                                 int(cluster_data['trigger_site']), str(parsed_word.word), site)
                    one_lexical_feature1 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                                 embeddings_length, cluster_data['trigger'],
                                                                 int(cluster_data['trigger_site']))
                    one_lexical_feature2 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
                                                                 embeddings_length, str(parsed_word.word), site)
                    if one_sentence_feature is not None and one_lexical_feature2 is not None and one_lexical_feature1 is not None:
                        negative_sentence_level.append(copy.deepcopy(one_sentence_feature))
                        # negative_sentence_between.append(copy.deepcopy(one_sentence_between))
                        # negative_sentence_after.append(copy.deepcopy(one_sentence_after))
                        negative_lexical_level.append(one_lexical_feature1 + one_lexical_feature2)
                site = site + len(parsed_word.word)
            # print("-------------单个聚类结果负例数据获取完毕-------------")

        # for cluster_data in negative_data:
        #     one_sentence_before, one_sentence_between, one_sentence_after \
        #         = sentence_feature_input(cluster_data['sentence'], embeddings_index, False, cluster_data['trigger'],
        #                                  cluster_data['trigger_site'], cluster_data[argument_type],
        #                                  cluster_data[argument_site])
        #     one_lexical_feature1 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
        #                                                  embeddings_length, cluster_data['trigger'],
        #                                                  cluster_data['trigger_site'])
        #     one_lexical_feature2 = lexical_level_feature(cluster_data['sentence'], embeddings_index,
        #                                                  embeddings_length, cluster_data[argument_type],
        #                                                  cluster_data[argument_site])
        #     if one_sentence_before is not None and one_lexical_feature2 is not None and one_lexical_feature1 is not None:
        #         negative_sentence_before.append(copy.deepcopy(one_sentence_before))
        #         negative_sentence_between.append(copy.deepcopy(one_sentence_between))
        #         negative_sentence_after.append(copy.deepcopy(one_sentence_after))
        #         negative_lexical_level.append(one_lexical_feature1 + one_lexical_feature2)

    # sentence_before = positive_sentence_before + negative_sentence_before
    # sentence_between = positive_sentence_between + negative_sentence_between
    # sentence_after = positive_sentence_after + negative_sentence_after
    sentence_level = positive_sentence_level + negative_sentence_level
    lexical_level = positive_lexical_level + negative_lexical_level
    label = [1] * len(positive_sentence_level) + [0] * len(negative_sentence_level)
    data_length = len(sentence_level)
    print("聚类结果中正例数量：", len(positive_sentence_level))
    print("聚类结果中负例数量：", len(negative_sentence_level))
    # 对数据做shuffle
    index = [i for i in range(data_length)]
    random.shuffle(index)
    sentence_level = [sentence_level[i] for i in index]
    # sentence_between = [sentence_between[i] for i in index]
    # sentence_after = [sentence_after[i] for i in index]
    lexical_level = [lexical_level[i] for i in index]
    label = [label[i] for i in index]

    # 对shape进行一定的调整
    sentence_level = np.asarray(sentence_level, dtype='float32')
    sentence_level = np.reshape(sentence_level, (data_length, 110, 302, 1))
    # sentence_between = np.asarray(sentence_between, dtype='float32')
    # sentence_between = np.reshape(sentence_between, (data_length, 110, 302, 1))
    # sentence_after = np.asarray(sentence_after, dtype='float32')
    # sentence_after = np.reshape(sentence_after, (data_length, 110, 302, 1))
    lexical_level = np.asarray(lexical_level, dtype='float32')
    label = np.asarray(label)
    print(argument_type, "训练数据长度:", data_length)
    return sentence_level,  lexical_level, label
    # return sentence_before, sentence_between, sentence_after, lexical_level, label


# 返回一个句子的句子级别的特征
def sentence_feature_input(sentence, embeddings_index, is_trigger,
                           trigger, trigger_site, argument=None, argument_site=None):
    # 句子的embedding表示
    cwf = []
    pf = []
    pf2 = []
    trigger_site = int(trigger_site)
    # 首先对句子进行分词
    sentence_split = jieba.lcut(sentence.strip('\n'))
    # 控制句子的长度在110
    if len(sentence_split) > 110:
        return None
        # if is_trigger is True:
        #     return None, None
        # else:
        #     return None, None, None
    elif len(sentence_split) < 110:
        for i in range(0, 110-len(sentence_split)):
            cwf.append([0]*300)
            pf.append(0)
            pf2.append(0)
    # print(sentence_split)
    trigger_num = -1
    argument_num = -1
    site = 0
    # 添加句子的cwf特征
    for index, word in enumerate(sentence_split):
        if trigger == word and site == trigger_site:
            trigger_num = index
        site = site + len(word)
        try:
            cwf.append(copy.deepcopy(embeddings_index[word]))
        except:
            cwf.append([0]*300)
    # cwf = keras_get_bert_embedding(sentence)
    # 如果trigger不在分词结果内，则忽略该句子
    if trigger_num == -1:
        # print("trigger不在分词结果内！\n", trigger, trigger_site, sentence_split)
        return None
        # if is_trigger is True:
        #     return None, None
        # else:
        #     return None, None, None
    site = 0
    # 如果不是trigger的特征，判断argument是否在分词结果内
    if not is_trigger:
        for index, word in enumerate(sentence_split):
            if argument == word and site == argument_site:
                argument_num = index
            site = site + len(word)
        if argument_num == -1:
            # print("argument不在分词结果内！\n", argument, argument_site, sentence_split)
            return None
            # return None, None, None
    res = copy.deepcopy(cwf)
    # 清理内存
    # del cwf
    # 添加句子的pf特征
    for index in range(len(sentence_split)):
        pf.append(index-trigger_num)
    # 如果为argument，还需要添加argument的位置信息
    if is_trigger is False:
        for index in range(len(sentence_split)):
            pf2.append(index-argument_num)
        for index, buf in enumerate(res):
            buf.append(pf[index])
            buf.append(pf2[index])
            res[index] = buf
    else:
        for index in range(len(res)):
            res[index].append(pf[index])
    # del pf
    # del pf2
    # 暂时不考虑事件的编码
    # ef = []
    # print(np.array(res).shape)
    # for row in res:
    #     print(row)
    #     print(len(row))

    # # 如果是trigger的特征，则根据trigger分割句子，并补长至110
    # if is_trigger is True:
    #     res_before_trigger = res[:trigger_num]
    #     res_after_trigger = res[trigger_num:]
    #     for i in range(110-len(res_before_trigger)):
    #         res_before_trigger.append([0] * 301)
    #     for i in range(110-len(res_after_trigger)):
    #         res_after_trigger.append([0] * 301)
    #     # print(np.asarray(res_before_trigger).shape)
    #     # print(np.asarray(res_after_trigger).shape)
    #     # del res
    #     return res_after_trigger, res_after_trigger
    # # 如果是argument的特征，则根据trigger和argument分割句子，并补长至110
    # else:
    #     if trigger_num < argument_num:
    #         min_num = trigger_num
    #         max_num = argument_num
    #     else:
    #         min_num = argument_num
    #         max_num = trigger_num
    #     res_before = res[:min_num]
    #     res_between = res[min_num:max_num]
    #     res_after = res[max_num:]
    #     for i in range(110-len(res_before)):
    #         res_before.append([0] * 302)
    #     for i in range(110-len(res_between)):
    #         res_between.append([0] * 302)
    #     for i in range(110-len(res_after)):
    #         res_after.append([0] * 302)
        # del res
        # print(np.asarray(res_before).shape)
        # print(np.asarray(res_between).shape)
        # print(np.asarray(res_after).shape)
    # return res_before, res_between, res_after
    return res


# 返回一个句子的词汇级别的特征
def lexical_level_feature(sentence, embeddings_index, embeddings_length, key_word, key_word_site):
    key_word_site = int(key_word_site)
    sentence_split = jieba.lcut(sentence.strip('\n'))
    if len(sentence_split) > 110:
        return None
    num = -1
    site = 0
    for index, word in enumerate(sentence_split):
        if word == key_word and site == key_word_site:
            num = index
            break
        site = site + len(word)
    if num == -1:
        return None
    try:
        buf2 = copy.deepcopy(embeddings_index[sentence_split[num]])
    except:
        buf2 = [0] * 300
    if num is 0:
        buf1 = [0] * int(embeddings_length)
        try:
            buf3 = copy.deepcopy(embeddings_index[sentence_split[num+1]])
        except:
            buf3 = [0] * 300
    elif num is (len(sentence_split) - 1):
        try:
            buf1 = copy.deepcopy(embeddings_index[sentence_split[num-1]])
        except:
            buf1 = [0] * 300
        buf3 = [0] * int(embeddings_length)
    else:
        try:
            buf3 = copy.deepcopy(embeddings_index[sentence_split[num+1]])
        except:
            buf3 = [0] * 300
        try:
            buf1 = copy.deepcopy(embeddings_index[sentence_split[num-1]])
        except:
            buf1 = [0] * 300
    buf1.extend(buf2)
    buf1.extend(buf3)
    # del sentence_split
    # del buf2
    # del buf3
    return buf1


def get_chinese_embedding(glove_dir='word_embedding/sgns.sogounews.bigram-char/sgns.sogounews.bigram-char'):
    f = open(glove_dir, "r", encoding="utf-8")
    # 获取词向量的维度,l表示单词数，w为某个单词转化为词向量后的维度
    l, w = f.readline().split()
    # print(l)
    # 创建词向量索引字典
    embeddings_index = {}
    for index, line in enumerate(f):
        # 读取词向量文件中的每一行
        # 过滤掉杂质
        values = line.split()
        if len(values) != int(w)+1:
            continue
        # 获取当前行的词
        word = values[0]
        # 获取当前词的词向量
        try:
            # coefs = np.asarray(values[1:], dtype="float32")
            coefs = values[1:]
            # 将读入的这行词向量加入词向量索引字典
            embeddings_index[word] = coefs
        except:
            continue
    f.close()
    # print(len(embeddings_index), len(embeddings_index['火箭']))
    return embeddings_index, w


if __name__ == '__main__':
    # cwf = sentence_feature_input("火箭以110-95战胜勇士，取得胜利！", embedding, False, "战胜", 9, "火箭", 0)
    # lexical_level_feature("火箭以110-95战胜勇士，取得胜利！", embedding, length, "战胜", 9)
    # trigger_data()
    corpus_sentence()
