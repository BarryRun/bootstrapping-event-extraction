
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import network
import json
import tensorflow as tf
from keras.models import load_model
import keras.backend.tensorflow_backend as ktf
config = tf.ConfigProto()
# 不全部占满显存, 按需分配
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

ktf.set_session(sess)

from spider import data_process
import copy
import dataProcess
import k_means
import numpy as np
import jieba
jieba.load_userdict('userDict.txt')
import argument_alignment
from pyhanlp import HanLP


def argument_evaluate(evaluate_file_path, winner_net_file, loser_net_file, winner_score_net_file, loser_score_net_file):
    embeddings_index, embeddings_length = dataProcess.get_chinese_embedding()
    print("加载词向量完成")
    with open(evaluate_file_path, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        evaluate_data = json.loads(json_str)
    print("加载测试数据完成")
    winner_net = load_model(winner_net_file)
    loser_net = load_model(loser_net_file)
    winner_score_net = load_model(winner_score_net_file)
    loser_score_net = load_model(loser_score_net_file)
    winner_TP = winner_TN = winner_FP = winner_FN = 0
    loser_TP = loser_TN = loser_FP = loser_FN = 0
    winner_score_TP = winner_score_TN = winner_score_FP = winner_score_FN = 0
    loser_score_TP = loser_score_TN = loser_score_FP = loser_score_FN = 0

    for index, data in enumerate(evaluate_data):
        print(index, data['sentence'])
        if index > 95:
            break
        before_length = 0
        winner_res_max = 0
        loser_res_max = 0
        winner_score_res_max = 0
        loser_score_res_max = 0
        team_num = 0
        score_num = 0
        winner_site = loser_site = winner_score_site = loser_score_site = -1
        winner = loser = winner_score = loser_score = ""
        sentence = data['sentence']
        trigger = data['trigger']
        trigger_site = data['trigger_site']
        sentence_split = jieba.lcut(sentence)
        # 对于每个存在触发词的句子，寻找其事件论元
        for word in sentence_split:
            one_sentence_level = \
                dataProcess.sentence_feature_input(sentence, embeddings_index, False, trigger, trigger_site,
                                                   word, before_length)
            one_lexical_feature1 = dataProcess.lexical_level_feature(sentence, embeddings_index,
                                                                     embeddings_length, trigger,
                                                                     trigger_site)
            one_lexical_feature2 = dataProcess.lexical_level_feature(sentence, embeddings_index,
                                                                     embeddings_length, word,
                                                                     before_length)
            if one_sentence_level is not None and one_lexical_feature1 is not None \
                    and one_lexical_feature2 is not None:
                # all_sentence_data[name].append(one_sentence_feature)
                one_sentence_level = np.asarray(one_sentence_level, dtype='float32').reshape((1, 110, 302, 1))
                # one_sentence_between = np.asarray(one_sentence_between, dtype='float32').reshape(
                #     (1, 110, 302, 1))
                # one_sentence_after = np.asarray(one_sentence_level, dtype='float32').reshape((1, 110, 302, 1))
                one_lexical_feature = np.asarray(one_lexical_feature1 + one_lexical_feature2,
                                                 dtype='float32').reshape((1, 1800))
                word_parse = HanLP.segment(word)
                if len(word_parse) > 1:
                    before_length = before_length + len(word)
                    continue
                word_nature = str(word_parse[0].nature)
                # 如果是名词，则判断是否为team论元
                if word_nature[0] == 'n':
                    # team_num += 1
                    winner_res = winner_net.predict(
                        x=[one_sentence_level, one_lexical_feature])
                    loser_res = loser_net.predict(
                        x=[one_sentence_level, one_lexical_feature])
                    # if winner_res > winner_res_max:
                    #     winner_site = before_length
                    #     winner = word
                    #     winner_res_max = winner_res
                    if winner_res > 0.5:
                        if before_length == int(data['winner_site']):
                            winner_TP += 1
                        else:
                            winner_FP += 1
                    else:
                        if before_length == int(data['winner_site']):
                            winner_FN += 1
                        else:
                            winner_TN += 1

                    if loser_res > 0.5:
                        if before_length == int(data['loser_site']):
                            loser_TP += 1
                        else:
                            loser_FP += 1
                    else:
                        if before_length == int(data['loser_site']):
                            loser_FN += 1
                        else:
                            loser_TN += 1

                # 如果是数词，则判断是否为score论元
                elif word_nature == 'm':
                    # score_num += 1
                    winner_score_res = winner_score_net.predict(
                        x=[one_sentence_level, one_lexical_feature])
                    loser_score_res = loser_score_net.predict(
                        x=[one_sentence_level, one_lexical_feature])
                    if winner_score_res > 0.5:
                        if before_length == int(data['winner_score_site']):
                            winner_score_TP += 1
                        else:
                            winner_score_FP += 1
                    else:
                        if before_length == int(data['winner_score_site']):
                            winner_score_FN += 1
                        else:
                            winner_score_TN += 1

                    if loser_score_res > 0.5:
                        if before_length == int(data['loser_score_site']):
                            loser_score_TP += 1
                        else:
                            loser_score_FP += 1
                    else:
                        if before_length == int(data['loser_score_site']):
                            loser_score_FN += 1
                        else:
                            loser_score_TN += 1
                    # if loser_score_res > loser_score_res_max:
                    #     loser_score_site = before_length
                    #     loser_score = word
                    #     loser_score_res_max = loser_score_res
            before_length = before_length + len(word)
        # if winner_site == data['winner_site'] and winner == data['winner']:
        #     winner_TP += 1
        #     winner_TN += team_num - 1
        # else:
        #     winner_FN += 1
        #     winner_FP += 1
        #     winner_TN += team_num - 1
        #
        # if loser_site == data['loser_site'] and loser == data['loser']:
        #     loser_TP += 1
        #     loser_TN += team_num - 1
        # else:
        #     loser_FN += 1
        #     loser_FP += 1
        #     loser_TN += team_num - 1
        #
        # if winner_score_site == data['winner_score_site'] and winner_score == data['winner_score']:
        #     winner_score_TP += 1
        #     winner_score_TN += score_num - 1
        # else:
        #     winner_score_FN += 1
        #     winner_score_FP += 1
        #     winner_score_TN += score_num - 1
        #
        # if loser_score_site == data['loser_score_site'] and loser_score == data['loser_score']:
        #     loser_score_TP += 1
        #     loser_score_TN += score_num - 1
        # else:
        #     loser_score_FN += 1
        #     loser_score_FP += 1
        #     loser_score_TN += score_num - 1

    print("Winner预测结果:")
    print("TP:", winner_TP)
    print("TN:", winner_TN)
    print("FP:", winner_FP)
    print("FN:", winner_FN)
    print("准确率：", (winner_TP + winner_TN)/(winner_TP + winner_TN + winner_FP + winner_FN))
    print("精确率：", winner_TP/(winner_TP + winner_FP))
    print("召回率：", winner_TP/(winner_TP + winner_FN))
    print("F1值：", 2 * winner_TP / (2 * winner_TP + winner_FP + winner_FN))

    print("Loser预测结果:")
    print("TP:", loser_TP)
    print("TN:", loser_TN)
    print("FP:", loser_FP)
    print("FN:", loser_FN)
    print("准确率：", (loser_TP + loser_TN)/(loser_TP + loser_TN + loser_FP + loser_FN))
    print("精确率：", loser_TP/(loser_TP + loser_FP))
    print("召回率：", loser_TP/(loser_TP + loser_FN))
    print("F1值：", 2 * loser_TP / (2 * loser_TP + loser_FP + loser_FN))

    print("WinnerScore预测结果:")
    print("TP:", winner_score_TP)
    print("TN:", winner_score_TN)
    print("FP:", winner_score_FP)
    print("FN:", winner_score_FN)
    print("准确率：", (winner_score_TP + winner_score_TN)/(winner_score_TP + winner_score_TN + winner_score_FP + winner_score_FN))
    print("精确率：", winner_score_TP/(winner_score_TP + winner_score_FP))
    print("召回率：", winner_score_TP/(winner_score_TP + winner_score_FN))
    print("F1值：", 2 * winner_score_TP / (2 * winner_score_TP + winner_score_FP + winner_score_FN))

    print("LoserScore预测结果:")
    print("TP:", loser_score_TP)
    print("TN:", loser_score_TN)
    print("FP:", loser_score_FP)
    print("FN:", loser_score_FN)
    print("准确率：", (loser_score_TP + loser_score_TN)/(loser_score_TP + loser_score_TN + loser_score_FP + loser_score_FN))
    print("精确率：", loser_score_TP/(loser_score_TP + loser_score_FP))
    print("召回率：", loser_score_TP/(loser_score_TP + loser_score_FN))
    print("F1值：", 2 * loser_score_TP / (2 * loser_score_TP + loser_score_FP + loser_score_FN))


def trigger_evaluate(evaluate_file_path, trigger_net_file):
    trained_net = load_model(trigger_net_file)
    embeddings_index, embeddings_length = dataProcess.get_chinese_embedding()
    with open(evaluate_file_path, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        evaluate_data = json.loads(json_str)
    trigger_TP = trigger_TN = trigger_FP = trigger_FN = 0

    for index, data in enumerate(evaluate_data):
        sentence = data['sentence']
        print(index, sentence)
        sentence_parsed = HanLP.segment(sentence)
        before_length = 0
        max_trigger_score = 0
        trigger_site = -1
        trigger = ""
        for index_word, parsed_word in enumerate(sentence_parsed):
            word = str(parsed_word.word)
            nature = str(parsed_word.nature)
            if str(nature) == 'v':
                one_sentence_level = dataProcess.sentence_feature_input(sentence,
                                                                         embeddings_index, True,
                                                                         str(word),
                                                                         before_length)
                one_lexical_feature = dataProcess.lexical_level_feature(sentence, embeddings_index,
                                                                        embeddings_length,
                                                                        str(word), before_length)
                if one_sentence_level is not None and one_lexical_feature is not None:
                    # all_sentence_data[name].append(one_sentence_feature)
                    one_sentence_level = np.asarray(one_sentence_level, dtype='float32').reshape((1, 110, 301, 1))
                    # one_sentence_after = np.asarray(one_sentence_after, dtype='float32').reshape((1, 110, 301, 1))
                    one_lexical_feature = np.asarray(one_lexical_feature, dtype='float32').reshape((1, 900))
                    res = trained_net.predict(x=[one_sentence_level, one_lexical_feature])
                    if index >= 96:
                        if before_length == data['trigger_site']:
                            if res > 0.5:
                                trigger_FP += 1
                            else:
                                trigger_TN += 1
                    else:
                        if res > 0.5:
                            if res > max_trigger_score:
                                max_trigger_score = res
                                trigger_site = before_length
                                trigger = word
            before_length = before_length + len(str(word))
        if index < 96:
            if trigger_site == int(data['trigger_site']) and trigger == data['trigger']:
                trigger_TP += 1
            else:
                trigger_FN += 1
        else:
            if trigger_site != int(data['trigger_site']) or trigger != data['trigger']:
                trigger_TN += 1
            else:
                trigger_FP += 1
    print("trigger预测结果:")
    print("TP:", trigger_TP)
    print("TN:", trigger_TN)
    print("FP:", trigger_FP)
    print("FN:", trigger_FN)
    print("准确率：", (trigger_TP + trigger_TN)/(trigger_TP + trigger_TN + trigger_FP + trigger_FN))
    print("精确率：", trigger_TP/(trigger_TP + trigger_FP))
    print("召回率：", trigger_TP/(trigger_TP + trigger_FN))
    print("F1值：", 2 * trigger_TP / (2 * trigger_TP + trigger_FP + trigger_FN))


# 初始种子数据
def initial_seed():
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    all_label_argument = {}
    # 读取所有标注，并以字典的形式存储
    f = open('第一轮迭代/seed_data.txt', 'w')
    for ann in f_ann:
        one_label = {}
        buf = ann.split()
        if buf[0][0] == 'T':
            # one_label['index'] = buf[0]
            one_label['type'] = buf[1]
            one_label['begin_site'] = buf[2]
            one_label['end_site'] = buf[3]
            one_label['content'] = buf[4]
            one_label['sentence_index'] = buf[5]
            sentence = f_txt[int(one_label['sentence_index'])].strip('\n')
            for (file_name, file_data) in zip(data_name, data_list):
                for index, one_sentence in enumerate(file_data):
                    if sentence in one_sentence:
                        # 找出对应句子所在的语料库
                        one_label['file_name'] = file_name
                        one_label['sentence'] = sentence
                        one_label['file_index'] = index
        elif buf[0][0] == 'E':
            # one_label['index'] = buf[0]
            for i in range(1, len(buf)):
                arguments = buf[i].split(':')
                one_label[arguments[0]] = arguments[1]
        all_label_argument[str(buf[0])] = copy.deepcopy(one_label)

    for key in all_label_argument.keys():
        one_label = all_label_argument[key]
        if key[0] == 'E' and len(one_label) == 5:
            print(str(one_label))
            one_data = dict()
            one_data['trigger'] = all_label_argument[one_label['Game']]['content']
            one_data['winner'] = all_label_argument[one_label['Winner']]['content']
            one_data['loser'] = all_label_argument[one_label['Loser']]['content']
            one_data['winner_score'] = all_label_argument[one_label['WinnerScore']]['content']
            one_data['loser_score'] = all_label_argument[one_label['LoserScore']]['content']
            one_data['trigger_site'] = all_label_argument[one_label['Game']]['begin_site']
            one_data['winner_site'] = all_label_argument[one_label['Winner']]['begin_site']
            one_data['loser_site'] = all_label_argument[one_label['Loser']]['begin_site']
            one_data['winner_score_site'] = all_label_argument[one_label['WinnerScore']]['begin_site']
            one_data['loser_score_site'] = all_label_argument[one_label['LoserScore']]['begin_site']
            one_data['file_name'] = all_label_argument[one_label['LoserScore']]['file_name']
            one_data['file_index'] = all_label_argument[one_label['LoserScore']]['file_index']
            one_data['sentence'] = all_label_argument[one_label['LoserScore']]['sentence']
            print(str(one_data))
            f.write(str(one_data) + '\n')
    f.close()


# 读取种子数据
def read_seed(_positive_seed_file, _negative_seed_file):
    # 读取种子数据，并以字典的形式存储
    with open(_positive_seed_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        positive_data = json.loads(json_str)
    with open(_negative_seed_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        negative_data = json.loads(json_str)
    return positive_data, negative_data


# 针对聚类结果，更新种子数据
def updata_seed(cluster_res, old_positive_seed_res_file, old_negative_seed_res_file, new_positive_seed_res_file,
                new_negative_seed_res_file):
    k_means.show_result(cluster_res, 4)
    positive_data, negative_data = k_means.score_to_cluster(old_positive_seed_res_file, cluster_res)
    for data in positive_data:
        data.pop('cluster')
    for data in negative_data:
        data.pop('cluster')
    # 读取种子数据，并以字典的形式存储
    with open(old_positive_seed_res_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        old_positive_data = json.loads(json_str)
    with open(old_negative_seed_res_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        old_negative_data = json.loads(json_str)
    positive_data = remove_repeat(positive_data, old_positive_data)
    negative_data = remove_repeat(negative_data, old_negative_data)
    with open(new_positive_seed_res_file, 'w') as f:
        for data in positive_data:
            f.write(str(data) + '\n')
    with open(new_negative_seed_res_file, 'w') as f:
        for data in negative_data:
            f.write(str(data) + '\n')


# 返回不重复的种子数据
def remove_repeat(new_data, ole_data):
    return_data = ole_data
    repeat = False
    for new in new_data:
        for old in ole_data:
            if new['sentence'] == old['sentence']:
                repeat = True
                break
        if not repeat:
            return_data.append(new)
        else:
            repeat = False
    return return_data


def argument_train(_positive_cluster_data, _negative_cluster_data, winner_model, loser_model,
                   winner_score_model, loser_score_model, which_argument=None):
    if which_argument == 'Winner':
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'Winner')
        network.argument_train(sentence_level,
                               lexical_level, label, winner_model)
    elif which_argument == "Loser":
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'Loser')
        network.argument_train(sentence_level,
                               lexical_level, label, loser_model)
    elif which_argument == "WinnerScore":
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'WinnerScore')
        network.argument_train(sentence_level,
                               lexical_level, label, winner_score_model)
    elif which_argument == "LoserScore":
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'LoserScore')
        network.argument_train(sentence_level,
                               lexical_level, label, loser_score_model)
    else:
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'Winner')
        network.argument_train(sentence_level,
                               lexical_level, label, winner_model)
        print("Winner Over")
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'Loser')
        network.argument_train(sentence_level,
                               lexical_level, label, loser_model)
        print("Loser Over")
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'WinnerScore')
        network.argument_train(sentence_level,
                               lexical_level, label, winner_score_model)
        print("WinnerScore Over")
        sentence_level, lexical_level, label = \
            dataProcess.argument_data(_positive_cluster_data, _negative_cluster_data, 'LoserScore')
        network.argument_train(sentence_level,
                               lexical_level, label, loser_score_model)
        print("LoserScore Over")


def argument_predict():
    network.argument_predict('第二轮迭代_非平行/winner_network.h5', '第二轮迭代_非平行/loser_network.h5',
                             '第二轮迭代_非平行/winner_score_network.h5', '第二轮迭代_非平行/loser_score_network.h5',
                             "第二轮迭代_非平行/trigger_predict_res.txt", "第二轮迭代_非平行/argument_predict_res.txt")


def trigger_train(_positive_cluster_data, _negative_cluster_data, network_save_path, res_save_path):
    sentence_level, lexical_level, label = dataProcess.trigger_data(_positive_cluster_data, _negative_cluster_data)
    network.trigger_train(sentence_level, lexical_level, label, network_save_path, res_save_path)


def train_all_argument():
    positive_data, negative_data = read_seed("第一轮迭代/positive_seed_data.txt", "第一轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第一轮迭代/winner_network.h5", '第一轮迭代/loser_network.h5',
                   '第一轮迭代/winner_score_network.h5', '第一轮迭代/loser_score_network.h5')
    print("第一轮结束")
    positive_data, negative_data = read_seed("第二轮迭代/positive_seed_data.txt", "第二轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第二轮迭代/winner_network.h5", '第二轮迭代/loser_network.h5',
                   '第二轮迭代/winner_score_network.h5', '第二轮迭代/loser_score_network.h5')
    print("第二轮结束")
    positive_data, negative_data = read_seed("第三轮迭代/positive_seed_data.txt", "第三轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第三轮迭代/winner_network.h5", '第三轮迭代/loser_network.h5',
                   '第三轮迭代/winner_score_network.h5', '第三轮迭代/loser_score_network.h5')
    print("第三轮结束")
    positive_data, negative_data = read_seed("第四轮迭代/positive_seed_data.txt", "第四轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第四轮迭代/winner_network.h5", '第四轮迭代/loser_network.h5',
                   '第四轮迭代/winner_score_network.h5', '第四轮迭代/loser_score_network.h5')
    print("第四轮结束")
    positive_data, negative_data = read_seed("第五轮迭代/positive_seed_data.txt", "第五轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第五轮迭代/winner_network.h5", '第五轮迭代/loser_network.h5',
                   '第五轮迭代/winner_score_network.h5', '第五轮迭代/loser_score_network.h5')


def get_good_result():
    positive_data, negative_data = read_seed("第五轮迭代/positive_seed_data.txt", "第五轮迭代/negative_seed_data.txt")
    argument_train(positive_data, negative_data, "第五轮迭代/winner_network.h5", '第五轮迭代/loser_network.h5',
                   '第五轮迭代/winner_score_network.h5', '第五轮迭代/loser_score_network.h5', which_argument="WinnerScore")
    argument_train(positive_data, negative_data, "第五轮迭代/winner_network.h5", '第五轮迭代/loser_network.h5',
                   '第五轮迭代/winner_score_network.h5', '第五轮迭代/loser_score_network.h5', which_argument="Winner")
    argument_evaluate("evaluate_data.txt", '第五轮迭代/winner_network.h5', '第五轮迭代/loser_network.h5',
                      '第五轮迭代/winner_score_network.h5', '第五轮迭代/loser_score_network.h5')


def evaluate_all_argument():
    argument_evaluate("evaluate_data.txt", '第一轮迭代/winner_network.h5', '第一轮迭代/loser_network.h5',
                      '第一轮迭代/winner_score_network.h5', '第一轮迭代/loser_score_network.h5')
    print("第一轮结束")
    argument_evaluate("evaluate_data.txt", '第二轮迭代/winner_network.h5', '第二轮迭代/loser_network.h5',
                      '第二轮迭代/winner_score_network.h5', '第二轮迭代/loser_score_network.h5')
    print("第二轮结束")
    argument_evaluate("evaluate_data.txt", '第三轮迭代/winner_network.h5', '第三轮迭代/loser_network.h5',
                      '第三轮迭代/winner_score_network.h5', '第三轮迭代/loser_score_network.h5')
    print("第三轮结束")
    argument_evaluate("evaluate_data.txt", '第四轮迭代/winner_network.h5', '第四轮迭代/loser_network.h5',
                      '第四轮迭代/winner_score_network.h5', '第四轮迭代/loser_score_network.h5')
    print("第四轮结束")
    argument_evaluate("evaluate_data.txt", '第五轮迭代/winner_network.h5', '第五轮迭代/loser_network.h5',
                      '第五轮迭代/winner_score_network.h5', '第五轮迭代/loser_score_network.h5')


if __name__ == '__main__':
    # get_good_result()
    # trigger_evaluate("evaluate_data.txt", "第三轮迭代_非平行/trigger_network.h5")
    argument_evaluate("evaluate_data.txt", '第三轮迭代_非平行/winner_network.h5', '第三轮迭代_非平行/loser_network.h5',
                      '第三轮迭代_非平行/winner_score_network.h5', '第三轮迭代_非平行/loser_score_network.h5')

    # positive_data, negative_data = read_seed("第三轮迭代_非平行/positive_seed_data.txt", "第三轮迭代_非平行/negative_seed_data.txt")
    # trigger_train(positive_data, negative_data, '第三轮迭代_非平行/trigger_network.h5', '第三轮迭代_非平行/trigger_predict_res.txt')
    # network.trigger_predict('第三轮迭代_非平行/trigger_network.h5', '第三轮迭代_非平行/trigger_predict_res.txt')
    # argument_train(positive_data, negative_data, "第三轮迭代_非平行/winner_network.h5", '第三轮迭代_非平行/loser_network.h5',
    #                '第三轮迭代_非平行/winner_score_network.h5', '第三轮迭代_非平行/loser_score_network.h5')
    # argument_predict()
    # argument_alignment.alignment_without_parallel("第二轮迭代_非平行/positive_seed_data.txt", '第二轮迭代_非平行/alignment_res.txt')
    # k_means.k_means_cluster('第二轮迭代_非平行/alignment_res.txt', '第二轮迭代_非平行/cluster_res.txt')
    # k_means.show_result('第二轮迭代_非平行/cluster_res.txt', 4)
    # updata_seed('第二轮迭代_非平行/cluster_res.txt', '第二轮迭代_非平行/positive_seed_data.txt',
    #             '第二轮迭代_非平行/negative_seed_data.txt', '第三轮迭代_非平行/positive_seed_data.txt',
    #             '第三轮迭代_非平行/negative_seed_data.txt')
    # positive_cluster_data, negative_cluster_data = k_means.score_to_cluster('第一轮迭代/positive_seed_data.txt',
    #                                                                         '第一轮迭代/cluster_res_team_and_score.txt')
    # positive_cluster_data, negative_cluster_data = k_means.score_to_cluster('第一轮迭代_非平行/positive_seed_data.txt',
    #                                                                         '第一轮迭代_非平行/cluster_res.txt')
