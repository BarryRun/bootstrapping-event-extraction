# -*- coding: UTF-8 -*-
from spider import data_process
from pyhanlp import HanLP
import copy
import jieba
import json
jieba.load_userdict('userDict.txt')


def alignment_without_parallel(argument_predict_res, alignment_res):
    # 1、获取已有的所有的argument，数据中应该包含文件名（语料库）
    # 2、根据已有的argument（队名或者分数），在所有语料库中来对齐获得其他包含它的句子（词性分析、句法分析等）
    # 3、返回所有获取到的句子
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    f = open(alignment_res, 'w')
    # 读取所有预测结果，并以字典的形式存储
    with open(argument_predict_res, 'r') as f2:
        json_str = '['
        for line in f2.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)

    # 避免重复，为所有已出现的句子打上已标注标签
    for one_data in data:
        content = one_data['sentence']
        for (file_name, file_data) in zip(data_name, data_list):
            for index, sentence in enumerate(file_data):
                if sentence == content:
                    print("已出现", sentence)
                    sentence = sentence + '已标注'
                    file_data[index] = sentence

    labeled = []
    for index_out, one_data in enumerate(data):
        print(index_out, "one_data", one_data)
        winner = one_data['winner']
        loser = one_data['loser']
        winner_score = one_data['winner_score']
        loser_score = one_data['loser_score']
        for (file_name, file_data) in zip(data_name, data_list):
            for index, sentence in enumerate(file_data):
                if sentence[-3:] == '已标注' or sentence in labeled:
                    # print("发现已标注：", sentence)
                    continue
                # 找出包含相同队伍和比分的句子
                sentence_split = jieba.lcut(sentence)
                one_aligned_sentence = dict()
                one_aligned_sentence['winner'] = winner
                one_aligned_sentence['loser'] = loser
                one_aligned_sentence['winner_score'] = winner_score
                one_aligned_sentence['loser_score'] = loser_score
                one_aligned_sentence['file_name'] = file_name
                one_aligned_sentence['file_index'] = index
                site = 0
                winner_site = -1
                loser_site = -1
                winner_score_site = -1
                loser_score_site = -1
                for word_index, word in enumerate(sentence_split):
                    if word == winner:
                        winner_site = site
                    if word == loser:
                        loser_site = site
                    if word == winner_score:
                        winner_score_site = site
                    if word == loser_score:
                        loser_score_site = site
                    site = site + len(word)
                # 如果四个argument都存在，则找出触发词进行保存
                # 此处可以根据需要修改，是否需要匹配所有的argument
                # if winner_site*loser_site*winner_score_site*loser_score_site != 0:
                if winner_site != -1 and loser_site != -1:
                    print("对齐到句子", sentence)
                    one_aligned_sentence['winner_site'] = winner_site
                    one_aligned_sentence['loser_site'] = loser_site
                    one_aligned_sentence['winner_score_site'] = winner_score_site
                    one_aligned_sentence['loser_score_site'] = loser_score_site
                    if winner_site < loser_site:
                        sentence_between = sentence[winner_site + len(winner): loser_site]
                        trigger_site = loser_site
                    else:
                        sentence_between = sentence[loser_site + len(loser): winner_site]
                        trigger_site = winner_site

                    # 这部分根据不同的问题需要进行修改
                    # 对句子做分词并进行词性分析
                    sentence_between_parsed = HanLP.segment(sentence_between)
                    sentence_between_parsed = list(sentence_between_parsed)
                    sentence_between_parsed.reverse()
                    for parsed_word in sentence_between_parsed:
                        # 找到两个队伍之间的最后一个动词作为触发词
                        trigger_site = trigger_site - len(str(parsed_word.word))
                        if str(parsed_word.nature) == 'v':
                            one_aligned_sentence['trigger'] = str(parsed_word.word)
                            one_aligned_sentence['trigger_site'] = trigger_site
                            print(one_aligned_sentence)
                            f.write(str(one_aligned_sentence) + '\n')
                            labeled.append(sentence)
                            break
    f.close()


def alignment_with_parallel(argument_predict_res, alignment_res):
    # 1、获取已有的所有的argument，数据中应该包含文件名（语料库）
    # 2、根据已有的argument（队名或者分数），在平行语料库中来对齐获得其他包含它的句子（词性分析、句法分析等）
    # 3、返回所有获取到的句子
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    f = open(alignment_res, 'w')
    # 读取所有预测结果，并以字典的形式存储
    with open(argument_predict_res, 'r') as f2:
        json_str = '['
        for line in f2.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)

    # 避免重复，为所有已出现的句子打上已标注标签
    for one_data in data:
        content = one_data['sentence']
        for (file_name, file_data) in zip(data_name, data_list):
            for index, sentence in enumerate(file_data):
                if sentence == content:
                    print("已出现", sentence)
                    sentence = sentence + '已标注'
                    file_data[index] = sentence

    labeled = []
    for index_out, one_data in enumerate(data):
        print(index_out, "one_data", one_data)
        winner = one_data['winner']
        loser = one_data['loser']
        winner_score = one_data['winner_score']
        loser_score = one_data['loser_score']
        for (file_name, file_data) in zip(data_name, data_list):
            for index, sentence in enumerate(file_data):
                if sentence[-3:] == '已标注' or sentence in labeled:
                    # print("发现已标注：", sentence)
                    continue
                # 找出包含相同队伍和比分的句子
                sentence_split = jieba.lcut(sentence)
                one_aligned_sentence = dict()
                one_aligned_sentence['winner'] = winner
                one_aligned_sentence['loser'] = loser
                one_aligned_sentence['winner_score'] = winner_score
                one_aligned_sentence['loser_score'] = loser_score
                one_aligned_sentence['file_name'] = file_name
                one_aligned_sentence['file_index'] = index
                site = 0
                winner_site = -1
                loser_site = -1
                winner_score_site = -1
                loser_score_site = -1
                for word_index, word in enumerate(sentence_split):
                    if word == winner:
                        winner_site = site
                    if word == loser:
                        loser_site = site
                    if word == winner_score:
                        winner_score_site = site
                    if word == loser_score:
                        loser_score_site = site
                    site = site + len(word)
                # 如果四个argument都存在，则找出触发词进行保存
                # 此处可以根据需要修改，是否需要匹配所有的argument
                # if winner_site*loser_site*winner_score_site*loser_score_site != 0:
                if winner_site != -1 and loser_site != -1 \
                        and winner_score_site != -1 and loser_score_site != -1:
                    print("对齐到句子", sentence)
                    one_aligned_sentence['winner_site'] = winner_site
                    one_aligned_sentence['loser_site'] = loser_site
                    one_aligned_sentence['winner_score_site'] = winner_score_site
                    one_aligned_sentence['loser_score_site'] = loser_score_site
                    if winner_site < loser_site:
                        sentence_between = sentence[winner_site + len(winner): loser_site]
                        trigger_site = loser_site
                    else:
                        sentence_between = sentence[loser_site + len(loser): winner_site]
                        trigger_site = winner_site

                    # 这部分根据不同的问题需要进行修改
                    # 对句子做分词并进行词性分析
                    sentence_between_parsed = HanLP.segment(sentence_between)
                    sentence_between_parsed = list(sentence_between_parsed)
                    sentence_between_parsed.reverse()
                    for parsed_word in sentence_between_parsed:
                        # 找到两个队伍之间的最后一个动词作为触发词
                        trigger_site = trigger_site - len(str(parsed_word.word))
                        if str(parsed_word.nature) == 'v':
                            one_aligned_sentence['trigger'] = str(parsed_word.word)
                            one_aligned_sentence['trigger_site'] = trigger_site
                            print(one_aligned_sentence)
                            f.write(str(one_aligned_sentence) + '\n')
                            labeled.append(sentence)
                            break
    f.close()


def argument_alignment(argument_predict_res):
    # 1、获取已有的所有的argument，数据中应该包含文件名（语料库）
    # 2、根据已有的argument（队名或者分数），来对齐获得其他包含它的句子（词性分析、句法分析等）
    # 3、返回所有获取到的句子
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    all_label_argument = {}
    f = open('第一轮迭代/alignment_res_team_and_score.txt', 'w')
    f2 = open(argument_predict_res, 'r')
    # 读取所有标注，并以字典的形式存储
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
                for one_sentence in file_data:
                    if sentence in one_sentence:
                        # 找出对应句子所在的语料库
                        one_label['corpus'] = file_name
        elif buf[0][0] == 'E':
            # one_label['index'] = buf[0]
            for i in range(1, len(buf)):
                arguments = buf[i].split(':')
                one_label[arguments[0]] = arguments[1]
        all_label_argument[str(buf[0])] = copy.deepcopy(one_label)

    for key in all_label_argument.keys():
        label_data = all_label_argument[key]
        # 对于标注数据中的每个正例数据
        if key[0] == 'E' and len(label_data) == 5:
            # file_data表示某个corpus_txt中的所有句子
            winner = all_label_argument[label_data['Winner']]['content']
            loser = all_label_argument[label_data['Loser']]['content']
            winner_score = all_label_argument[label_data['WinnerScore']]['content']
            loser_score = all_label_argument[label_data['LoserScore']]['content']
            content = f_txt[int(all_label_argument[label_data['Winner']]['sentence_index'])].strip('\n')
            # 此处并没有在平行语料库中做对齐
            # 如果是在平行语料库中对齐，可以使用到corpus对应值来筛选file_name
            for (file_name, file_data) in zip(data_name, data_list):
                for index, sentence in enumerate(file_data):
                    if sentence == content:
                        continue
                    # 找出包含相同队伍和比分的句子
                    sentence_split = jieba.lcut(sentence)
                    one_aligned_sentence = dict()
                    one_aligned_sentence['winner'] = winner
                    one_aligned_sentence['loser'] = loser
                    one_aligned_sentence['winner_score'] = winner_score
                    one_aligned_sentence['loser_score'] = loser_score
                    one_aligned_sentence['file_name'] = file_name
                    one_aligned_sentence['file_index'] = index
                    site = 0
                    winner_sites = []
                    loser_sites = []
                    winner_score_sites = []
                    loser_score_sites = []
                    for word_index, word in enumerate(sentence_split):
                        if word == winner:
                            winner_sites.append(site)
                        if word == loser:
                            loser_sites.append(site)
                        if word == winner_score:
                            winner_score_sites.append(site)
                        if word == loser_score:
                            loser_score_sites.append(site)
                        site = site + len(word)
                    # 如果四个argument都存在，则找出触发词进行保存
                    # 此处可以根据需要修改，是否需要匹配所有的argument
                    # if winner_site*loser_site*winner_score_site*loser_score_site != 0:
                    if len(winner_sites) != 0 and len(loser_sites) != 0 \
                            and len(winner_score_sites) != 0 and len(loser_score_sites) != 0:
                        for winner_site in winner_sites:
                            for loser_site in loser_sites:
                                for winner_score_site in winner_score_sites:
                                    for loser_score_site in loser_score_sites:
                                        one_aligned_sentence['winner_site'] = winner_site
                                        one_aligned_sentence['loser_site'] = loser_site
                                        one_aligned_sentence['winner_score_site'] = winner_score_site
                                        one_aligned_sentence['loser_score_site'] = loser_score_site
                                        print(key, sentence)
                                        if winner_site < loser_site:
                                            sentence_between = sentence[winner_site + len(winner): loser_site]
                                            trigger_site = loser_site
                                        else:
                                            sentence_between = sentence[loser_site + len(loser): winner_site]
                                            trigger_site = winner_site

                                        # 这部分根据不同的问题需要进行修改
                                        # 对句子做分词并进行词性分析
                                        sentence_between_parsed = HanLP.segment(sentence_between)
                                        sentence_between_parsed = list(sentence_between_parsed)
                                        sentence_between_parsed.reverse()
                                        for parsed_word in sentence_between_parsed:
                                            # 找到两个队伍之间的最后一个动词作为触发词
                                            trigger_site = trigger_site - len(str(parsed_word.word))
                                            if str(parsed_word.nature) == 'v':
                                                one_aligned_sentence['trigger'] = str(parsed_word.word)
                                                one_aligned_sentence['trigger_site'] = trigger_site
                                                print(one_aligned_sentence)
                                                f.write(str(one_aligned_sentence) + '\n')
                                                break


def argument_score_alignment():
    # 1、获取已有的所有的argument，数据中应该包含文件名（语料库）
    # 2、根据已有的argument（队名或者分数），来对齐获得其他包含它的句子（词性分析、句法分析等）
    # 3、返回所有获取到的句子
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    all_label_argument = {}
    f = open('第一轮迭代/alignment_res_only_score.txt', 'w')
    # 读取所有标注，并以字典的形式存储
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
                for one_sentence in file_data:
                    if sentence in one_sentence:
                        # 找出对应句子所在的语料库
                        one_label['corpus'] = file_name
        elif buf[0][0] == 'E':
            # one_label['index'] = buf[0]
            for i in range(1, len(buf)):
                arguments = buf[i].split(':')
                one_label[arguments[0]] = arguments[1]
        all_label_argument[str(buf[0])] = copy.deepcopy(one_label)
    aligned_sentence = []

    for key in all_label_argument.keys():
        label_data = all_label_argument[key]
        # 对于标注数据中的每个正例数据
        if key[0] == 'E' and len(label_data) == 5:
            # file_data表示某个corpus_txt中的所有句子
            winner = all_label_argument[label_data['Winner']]['content']
            loser = all_label_argument[label_data['Loser']]['content']
            winner_score = all_label_argument[label_data['WinnerScore']]['content']
            loser_score = all_label_argument[label_data['LoserScore']]['content']
            content = f_txt[int(all_label_argument[label_data['Winner']]['sentence_index'])].strip('\n')
            # 此处并没有在平行语料库中做对齐
            # 如果是在平行语料库中对齐，可以使用到corpus对应值来筛选file_name
            for (file_name, file_data) in zip(data_name, data_list):
                for index, sentence in enumerate(file_data):
                    if sentence == content:
                        continue
                    # 找出包含相同比分的句子
                    sentence_split = jieba.lcut(sentence)
                    one_aligned_sentence = dict()
                    one_aligned_sentence['winner'] = winner
                    one_aligned_sentence['loser'] = loser
                    one_aligned_sentence['winner_score'] = winner_score
                    one_aligned_sentence['loser_score'] = loser_score
                    one_aligned_sentence['file_name'] = file_name
                    one_aligned_sentence['file_index'] = index
                    site = winner_site = loser_site = winner_score_site = loser_score_site = -1
                    # 此处可以有讲究...因为一个句子内可能会出现多次
                    for word_index, word in enumerate(sentence_split):
                        if word == winner:
                            winner_site = site
                        if word == loser:
                            loser_site = site
                        if word == winner_score:
                            winner_score_site = site
                        if word == loser_score:
                            loser_score_site = site
                        site = site + len(word)
                    # 如果比分的argument都存在，则找出触发词进行保存
                    if winner_score_site != -1 and loser_score_site != -1:
                        one_aligned_sentence['winner_site'] = winner_site
                        one_aligned_sentence['loser_site'] = loser_site
                        one_aligned_sentence['winner_score_site'] = winner_score_site
                        one_aligned_sentence['loser_score_site'] = loser_score_site
                        print(key, sentence)
                        if winner_score_site < loser_score_site:
                            trigger_site = loser_score_site + len(loser_score)
                        else:
                            trigger_site = winner_score_site + len(winner_score)
                        sentence_after = sentence[trigger_site:]
                        # 这部分根据不同的问题需要进行修改
                        # 对句子做分词并进行词性分析
                        sentence_between_parsed = HanLP.segment(sentence_after)
                        sentence_between_parsed = list(sentence_between_parsed)
                        for parsed_word in sentence_between_parsed:
                            # 比分后面的第一个动词作为触发词
                            if str(parsed_word.nature) == 'v':
                                one_aligned_sentence['trigger'] = str(parsed_word.word)
                                one_aligned_sentence['trigger_site'] = trigger_site
                                print(one_aligned_sentence)
                                f.write(str(one_aligned_sentence) + '\n')
                                aligned_sentence.append(copy.deepcopy(one_aligned_sentence))
                                break
                            trigger_site = trigger_site + len(str(parsed_word.word))


if __name__ == '__main__':
    # res = HanLP.segment("76人主场战胜了独行侠")
    # print(res)
    alignment_without_parallel("第二轮迭代/argument_predict_res.txt")
