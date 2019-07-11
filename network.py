from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Conv2D, Input, MaxPool2D, Reshape, concatenate, Dense
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from spider import data_process
from pyhanlp import HanLP
import random
import gc
from memory_profiler import profile
import jieba
jieba.load_userdict('userDict.txt')
import dataProcess
import numpy as np
import copy
from keras import backend as bk


def trigger_model(sentence_length, window_length, filter_num, word_embedding_size,
                  position_embedding_size, event_type_embedding_size):
    sentence_feature_size = word_embedding_size + position_embedding_size + event_type_embedding_size
    lexical_features_size = 3 * word_embedding_size
    a1 = Input(shape=(sentence_length, sentence_feature_size, 1))
    # a2 = Input(shape=(sentence_length, sentence_feature_size, 1))
    b1 = Conv2D(filters=filter_num, kernel_size=(window_length, sentence_feature_size),
                strides=(1, 1), padding='valid', activation='relu')(a1)
    # b2 = Conv2D(filters=filter_num, kernel_size=(window_length, sentence_feature_size),
    #             strides=(1, 1), padding='valid', activation='relu')(a2)
    c1 = MaxPool2D(pool_size=(sentence_length-window_length+1, 1))(b1)
    # c2 = MaxPool2D(pool_size=(sentence_length-window_length+1, 1))(b2)
    d1 = Reshape((3, ))(c1)
    # d2 = Reshape((3, ))(c2)
    e = Input(shape=(lexical_features_size, ))
    f = concatenate([e, d1])
    # f = concatenate([e, d1, d2])
    res = Dense(1, activation='sigmoid')(f)
    # res = Reshape((1, ))(g)

    ''' 验证reshape的作用
    b = bk.variable(value=[[[[1]],[[9]],[[2]]]])
    print(bk.eval(b))
    d = bk.reshape(b, (1, 3))
    print(bk.eval(d))
    '''
    # _model = Model(inputs=[a1, a2, e], outputs=res)
    _model = Model(inputs=[a1, e], outputs=res)
    plot_model(_model, to_file='trigger_model.png', show_shapes=True)
    return _model


def argument_model(sentence_length, window_length, filter_num, word_embedding_size,
                   position_embedding_size, event_type_embedding_size):
    sentence_feature_size = word_embedding_size + position_embedding_size + event_type_embedding_size
    lexical_features_size = 3 * word_embedding_size
    a1 = Input(shape=(sentence_length, sentence_feature_size, 1))
    # a2 = Input(shape=(sentence_length, sentence_feature_size, 1))
    # a3 = Input(shape=(sentence_length, sentence_feature_size, 1))
    b1 = Conv2D(filters=filter_num, kernel_size=(window_length, sentence_feature_size),
                strides=(1, 1), padding='valid', activation='relu')(a1)
    # b2 = Conv2D(filters=filter_num, kernel_size=(window_length, sentence_feature_size),
    #             strides=(1, 1), padding='valid', activation='relu')(a2)
    # b3 = Conv2D(filters=filter_num, kernel_size=(window_length, sentence_feature_size),
    #             strides=(1, 1), padding='valid', activation='relu')(a3)
    c1 = MaxPool2D(pool_size=(sentence_length-window_length+1, 1))(b1)
    # c2 = MaxPool2D(pool_size=(sentence_length-window_length+1, 1))(b2)
    # c3 = MaxPool2D(pool_size=(sentence_length-window_length+1, 1))(b3)
    d1 = Reshape((3, ))(c1)
    # d2 = Reshape((3, ))(c2)
    # d3 = Reshape((3, ))(c3)
    e = Input(shape=(lexical_features_size * 2, ))
    f = concatenate([e, d1])
    # f = concatenate([e, d1, d2, d3])
    res = Dense(1, activation='sigmoid')(f)
    # res = Reshape((1, ))(g)

    ''' 验证reshape的作用
    b = bk.variable(value=[[[[1]],[[9]],[[2]]]])
    print(bk.eval(b))
    d = bk.reshape(b, (1, 3))
    print(bk.eval(d))
    '''
    # _model = Model(inputs=[a1, a2, a3, e], outputs=res)
    _model = Model(inputs=[a1, e], outputs=res)
    plot_model(_model, to_file='argument_model.png', show_shapes=True)
    return _model


# def trigger_train(sentence_before, sentence_after, lexical_level, label, model_saved_path, predict_saved_path):
def trigger_train(sentence_level, lexical_level, label, model_saved_path, predict_saved_path):
    net = trigger_model(sentence_length=110, window_length=3, filter_num=3,
                        word_embedding_size=300, position_embedding_size=1, event_type_embedding_size=0)
    data_length = len(sentence_level)
    print("训练数据长度", data_length)
    # sentence_level = np.asarray(sentence_level, dtype='float32')
    # lexical_level = np.asarray(lexical_level, dtype='float32')
    # label = np.asarray(label, dtype='float32')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, mode='min')
    checkpoint = ModelCheckpoint(model_saved_path, monitor='val_loss',  verbose=1, save_best_only=True, mode='min')
    net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    net.fit([sentence_level, lexical_level], label, epochs=30, batch_size=32, verbose=2,
            callbacks=[TensorBoard(log_dir='./log'), checkpoint, early_stopping], validation_split=0.2, shuffle=True)
    # net = load_model(model_saved_path)
    # trigger_predict(net, predict_saved_path)


# def argument_train(sentence_before, sentence_between, sentence_after,
def argument_train(sentence_level,
                   lexical_level, label, model_saved_path):
    net = argument_model(sentence_length=110, window_length=3, filter_num=3,
                         word_embedding_size=300, position_embedding_size=2, event_type_embedding_size=0)
    data_length = len(sentence_level)
    print("训练数据长度", data_length)
    # sentence_level = np.asarray(sentence_level, dtype='float32')
    # lexical_level = np.asarray(lexical_level, dtype='float32')
    # label = np.asarray(label, dtype='float32')
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, mode='min')
    checkpoint = ModelCheckpoint(model_saved_path, monitor='loss',  verbose=1, save_best_only=True, mode='min')
    net.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    net.fit([sentence_level, lexical_level], label, epochs=30, batch_size=32,
            verbose=2, callbacks=[TensorBoard(log_dir='./log'), checkpoint], shuffle=False, validation_split=0.2)
    # net = load_model(model_saved_path)
    # argument_predict(net, "第二轮迭代/trigger_predict_res.txt")


# @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
# @profile(precision=4)
def argument_predict(winner_net_file, loser_net_file, winner_score_net_file, loser_score_net_file,
                     trigger_res_file, predict_res_saved_path):
    embeddings_index, embeddings_length = dataProcess.get_chinese_embedding()
    print("词向量加载完毕")
    winner_net = load_model(winner_net_file)
    loser_net = load_model(loser_net_file)
    winner_score_net = load_model(winner_score_net_file)
    loser_score_net = load_model(loser_score_net_file)
    print("模型加载完毕")
    with open(trigger_res_file, 'r') as f:
        for index, line in enumerate(f):
            if index <= 741:
                print("trigger" + str(index) + " jump")
                continue
            line_split = line.strip('\n').split('\t')
            trigger = str(line_split[4])
            trigger_site = int(line_split[3])
            score = float(line_split[5][3:-2])
            sentence = line_split[1][1:]
            # print(score, trigger, sentence)
            if score > 0.91:
                sentence_split = jieba.lcut(sentence)
                print(score, trigger, sentence_split)
                # print(sentence_split)
                before_length = 0
                winner_res_max = 0
                loser_res_max = 0
                winner_score_res_max = 0
                loser_score_res_max = 0
                one_res = {}
                one_res['trigger'] = trigger
                one_res['trigger_site'] = trigger_site
                one_res['sentence'] = sentence
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
                        # one_sentence_after = np.asarray(one_sentence_after, dtype='float32').reshape((1, 110, 302, 1))
                        one_lexical_feature = np.asarray(one_lexical_feature1 + one_lexical_feature2,
                                                         dtype='float32').reshape((1, 1800))
                        word_parse = HanLP.segment(word)
                        if len(word_parse) > 1:
                            before_length = before_length + len(word)
                            continue
                        word_nature = str(word_parse[0].nature)
                        # 如果是名词，则判断是否为team论元
                        if word_nature[0] == 'n':
                            winner_res = winner_net.predict(x=[one_sentence_level, one_lexical_feature])
                            loser_res = loser_net.predict(x=[one_sentence_level, one_lexical_feature])
                            if winner_res > winner_res_max:
                                one_res['winner_site'] = before_length
                                one_res['winner'] = word
                                winner_res_max = winner_res
                            if loser_res > loser_res_max:
                                one_res['loser_site'] = before_length
                                one_res['loser'] = word
                                loser_res_max = loser_res
                        # 如果是数词，则判断是否为score论元
                        elif word_nature == 'm':
                            winner_score_res = winner_score_net.predict(x=[one_sentence_level, one_lexical_feature])
                            loser_score_res = loser_score_net.predict(x=[one_sentence_level, one_lexical_feature])
                            if winner_score_res > winner_score_res_max:
                                one_res['winner_score_site'] = before_length
                                one_res['winner_score'] = word
                                winner_score_res_max = winner_score_res
                            if loser_score_res > loser_score_res_max:
                                one_res['loser_score_site'] = before_length
                                one_res['loser_score'] = word
                                loser_score_res_max = loser_score_res
                    before_length = before_length + len(word)
                if len(one_res) == 11 and winner_res_max > 0.5 and loser_res_max > 0.5 and winner_score_res_max > 0.5\
                        and loser_score_res_max > 0.5:
                    print(str(one_res))
                    pre_res_file = open(predict_res_saved_path, 'a')
                    pre_res_file.write(str(one_res)+'\n')
                    pre_res_file.close()
                # gc.collect()
            print('trigger' + str(index) + ' over')


def trigger_predict(trained_net_path, predict_res_saved_path):
    trained_net = load_model(trained_net_path)
    embeddings_index, embeddings_length = dataProcess.get_chinese_embedding()
    file_name, file_data = data_process.read_text_from_corpus('spider/corpus_txt/')
    f = open(predict_res_saved_path, 'w')
    f.write('文件名\t句子\t句子序号\t词汇偏移量\t词汇\t预测结果\n')
    f.close()
    for file_index, (name, data) in enumerate(zip(file_name, file_data)):
        if file_index < 0:
            continue
        for sentence_index, sentence in enumerate(data):
            sentence_parsed = HanLP.segment(sentence)
            before_length = 0
            for index, parsed_word in enumerate(sentence_parsed):
                word = parsed_word.word
                nature = parsed_word.nature
                if str(nature) == 'v':
                    one_sentence_level = dataProcess.sentence_feature_input(sentence, embeddings_index, True,
                                                                            str(word), before_length)
                    one_lexical_feature = dataProcess.lexical_level_feature(sentence, embeddings_index,
                                                                            embeddings_length,
                                                                            str(word), before_length)
                    if one_sentence_level is not None and one_lexical_feature is not None:
                        # all_sentence_data[name].append(one_sentence_feature)
                        one_sentence_level = np.asarray(one_sentence_level, dtype='float32').reshape((1, 110, 301, 1))
                        one_lexical_feature = np.asarray(one_lexical_feature, dtype='float32').reshape((1, 900))
                        res = trained_net.predict(x=[one_sentence_level, one_lexical_feature])
                        print(str(file_index), str(sentence_index), str(index), str(word), str(res))
                        if res > 0.5:
                            f = open(predict_res_saved_path, 'a')
                            f.write(str(name) + '\t ' + sentence + '\t' + str(sentence_index) + '\t'
                                    + str(before_length) + '\t' + str(word) + '\t ' + str(res) + '\n')
                            f.close()
                before_length = before_length + len(str(word))


def argument_evaluate(argument_predict_res):
    with open(argument_predict_res, 'r') as f:
        buf = f.readlines()
        data_length = len(buf)
        index = [i for i in range(data_length)]
        random.shuffle(index)
        buf = [buf[index[i]] for i in range(50)]
    with open('第二轮迭代/50_argument_predict_res.txt', 'w') as f:
        for i in buf:
            f.write(i)


if __name__ == '__main__':
    # trigger_model(140, 3, 3, 300, 1, 0)
    # _net = argument_model(sentence_length=110, window_length=3, filter_num=3,
    #                       word_embedding_size=300, position_embedding_size=1, event_type_embedding_size=0)
    # argument_evaluate("第二轮迭代/argument_predict_res.txt")
    res = HanLP.segment('约基奇得到三双29分、11个篮板和10次助攻，穆雷得到18分，他们两人在关键时刻联手4分为球队锁定胜局，掘金队在客场以103-99险胜迈阿密热火队（19胜20负）。')
    print(res)
