from collections import defaultdict
from random import uniform
from math import sqrt
import numpy as np
import json
import jieba
import dataProcess
import copy
from spider import data_process
jieba.load_userdict('userDict.txt')


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2
    
    Returns a new point which is the center of all the points.
    """
    # 获取特征的维度
    dimensions = len(points[0])

    new_center = []

    # 计算每个维度上面所有点对应值的和
    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += float(p[dimension])

        # 计算每个维度上点的平均值
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.

    Compute the center for each of the assigned groups.

    Return `k` centers where `k` is the number of unique assignments.
    """
    # 定义一个字典，其默认初始化为一个空列表
    new_means = defaultdict(list)
    centers = []

    # 对于所有点以及其所属类别
    for assignment, point in zip(assignments, data_set):
        # 将不同类别的点分开
        new_means[assignment].append(point)

    # 对于每个类别的所有点
    for points in new_means.values():
        # 通过point_avg计算其中心点
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point. 
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []

    # 对于数据中的每一个点
    for point in data_points:
        # float("inf")表示正无穷，加上负号，即float("-inf")为负无穷
        shortest = float("inf")  # positive infinity
        shortest_index = 0
        # 对于每一个中心
        for i in range(len(centers)):
            # 计算到每一个中心的距离
            val = distance_trigger(point, centers[i])
            # val = distance(point, centers[i])

            # 记录距离最短的中心距离，以及其index
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    # 返回一个数组，对应data_points中相同index的点类别（即距离最近的中心点）
    return assignments


def distance_trigger(a, b):
    # 输入均为长度为900的一维向量
    # 每300维计算一个点乘,返回其中的最小值
    sum1 = sum2 = sum3 = 0
    for dimension in range(900):
        difference_sq = (float(a[dimension]) - float(b[dimension])) ** 2
        if dimension < 300:
            sum1 += difference_sq
        elif 300 <= dimension < 600:
            sum2 += difference_sq
        else:
            sum3 += difference_sq
    sum1 = sqrt(sum1)
    sum2 = sqrt(sum2)
    sum3 = sqrt(sum3)
    return min(sum1, sum2, sum3)


def distance(a, b):
    """
    """
    dimensions = len(a)
    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])   # 特征的维度
    min_max = defaultdict(float)  # defaultdict(int)会为新的key默认初始化为0

    # 对于数据集中的每一点
    for point in data_set:
        # 记录每一维度上的最大最小值
        for i in range(dimensions):
            val = float(point[i])
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    # 生成k个随机点
    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]
            # uniform方法随机生成一个范围内的实数
            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    # 返回k个随机点，数组(k, 900)
    return centers


def k_means(dataset, k):
    # 随机产生k个中心点
    k_points = generate_k(dataset, k)
    # 为每个点进行分类
    assignments = assign_points(dataset, k_points)

    old_assignments = None
    # for i in range(0, 3):
    while assignments != old_assignments:
        # 重新计算每个类别的中心点
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments

        # 为每个点进行分类
        assignments = assign_points(dataset, new_centers)
    return assignments


# 计算向量的余弦相似度
def cos(vector1, vector2):
    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vector1, vector2):
        a = float(a)
        b = float(b)
        dot_product += a*b
        norm_a += a**2
        norm_b += b**2
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    else:
        return dot_product / ((norm_a*norm_b)**0.5)


def similarity(feature1, feature2):
    # embedding1_1 = np.asarray(feature1[:300], dtype=float)
    # embedding1_2 = np.asarray(feature1[300:600], dtype=float)
    # embedding1_3 = np.asarray(feature1[600:], dtype=float)
    # embedding2_1 = np.asarray(feature2[:300], dtype=float)
    # embedding2_2 = np.asarray(feature2[300:600], dtype=float)
    # embedding2_3 = np.asarray(feature2[600:], dtype=float)
    # return np.dot(embedding1_1, embedding2_1) + np.dot(embedding1_2, embedding2_2) + np.dot(embedding1_3, embedding2_3)
    cos1 = cos(feature1[:300], feature2[:300])
    cos2 = cos(feature1[300:600], feature2[300:600])
    cos3 = cos(feature1[600:], feature2[600:])
    if cos1 is None or cos2 is None or cos3 is None:
        return None
    else:
        return max(cos1, cos2, cos3)


# 聚类所用的句子特征
def cluster_sentence_feature(sentence, trigger, trigger_site, embeddings_index):
    sentence_split = jieba.lcut(sentence)
    print(trigger)
    print(sentence_split)
    embedding_before_trigger = [0.0] * 300
    embedding_after_trigger = [0.0] * 300
    embedding_trigger = [0.0] * 300
    # print(sentence_split)
    # trigger_num = -1
    site = 0
    for index, word in enumerate(sentence_split):
        try:
            this_embedding = embeddings_index[word]
        except:
            this_embedding = [0] * 300

        # if trigger == word and site == trigger_site:
        #     trigger_num = index

        if site < trigger_site:
            embedding_before_trigger = [embedding_before_trigger[i] + float(this_embedding[i]) for i in
                                        range(300)]
        elif site == trigger_site:
            embedding_trigger = this_embedding
        else:
            embedding_after_trigger = [embedding_after_trigger[i] + float(this_embedding[i]) for i in
                                       range(300)]
        site = site + len(word)
    if embedding_trigger == [0.0]*300:
        embedding_trigger = embeddings_index[trigger]
    # assert trigger_num != -1
    embedding = embedding_before_trigger + embedding_trigger + embedding_after_trigger
    return embedding


# 评分所用的句子特征
def score_sentence_feature(sentence, trigger, trigger_site, embeddings_index):
    sentence_split = jieba.lcut(sentence)
    # print(trigger)
    # print(sentence_split)
    embedding_before_trigger = [0.0] * 300
    embedding_after_trigger = [0.0] * 300
    embedding_trigger = [0.0] * 300
    # print(sentence_split)
    # trigger_num = -1
    site = 0
    for index, word in enumerate(sentence_split):
        try:
            this_embedding = embeddings_index[word]
        except:
            this_embedding = [0] * 300

        # if trigger == word and site == trigger_site:
        #     trigger_num = index

        if site < trigger_site:
            embedding_before_trigger = [embedding_before_trigger[i] + float(this_embedding[i]) for i in
                                        range(300)]
        elif site == trigger_site:
            embedding_trigger = this_embedding
        else:
            embedding_after_trigger = [embedding_after_trigger[i] + float(this_embedding[i]) for i in
                                       range(300)]
        site = site + len(word)
    if embedding_trigger == [0.0]*300:
        embedding_trigger = embeddings_index[trigger]
    # assert trigger_num != -1
    embedding = embedding_before_trigger + embedding_trigger + embedding_after_trigger
    return embedding


# 计算各句子的表示,即针对触发词的聚类特征
def get_input(filename, embeddings_index):
    data_name, data_list = data_process.read_text_from_corpus('spider/corpus_txt/')
    # 通过json的方式读取txt文件中的字典
    with open(filename, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        print(json_str)
        data = json.loads(json_str)
    k_means_input = []
    sentences = []
    # 对于每一个读取到的数据
    for one_data in data:
        sentence = ''
        # 找到其对应的句子
        for (file_name, file_data) in zip(data_name, data_list):
            if file_name == one_data['file_name']:
                sentence = file_data[int(one_data['file_index'])]
                break
        # 如果找到了就获取该句子特征的表示（trigger暂定为3个embedding）
        if sentence != '':
            sentences.append(sentence)
            embedding = cluster_sentence_feature(sentence, one_data['trigger'], one_data['trigger_site'], embeddings_index)
            # cwf = keras_get_bert_embedding(sentence)
            # 如果trigger不在分词结果内，报错
            site = 0
            k_means_input.append(embedding)
    print(np.asarray(k_means_input).shape)
    return k_means_input, data, sentences


# 可视化聚类结果
def show_result(file_name, k):
    # 通过json的方式读取txt文件中的字典
    with open(file_name, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        data = json.loads(json_str)
    res = defaultdict(list)
    for one_data in data:
        res[one_data['cluster']].append((one_data['winner_site'], one_data['loser_site'], one_data['trigger'], one_data['sentence'],))
    for i in range(k):
        print(len(res[i]))
        for j in res[i]:
            print(i, str(j))


def score_to_cluster(positive_seed_file, cluster_res_file):
    # show_result(file_name, 4)
    # 对聚类每个类别进行评分
    # 评分的方式：对于聚类的每个数据，对每个种子数据做相似度计算，取最大值作为该数据的评分；
    #             每一类，计算所有数据评分的平均值，作为该数据的最终评分
    f_txt, f_ann = data_process.labeled_data_process('spider/dataExtracted.txt', 'spider/dataExtracted.ann')
    embeddings_index, embeddings_length = dataProcess.get_chinese_embedding()
    # all_label = {}
    # 读取种子数据，并以字典的形式存储
    with open(positive_seed_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        positive_data = json.loads(json_str)

    # 读取所有标注，并以字典的形式存储
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
    #         one_label['sentence'] = sentence
    #     elif buf[0][0] == 'E':
    #         for i in range(1, len(buf)):
    #             arguments = buf[i].split(':')
    #             one_label[arguments[0]] = arguments[1]
    #     all_label[str(buf[0])] = copy.deepcopy(one_label)

    # 针对种子数据每一个事件，得到其embedding表示
    event_embeddings = []
    for one_data in positive_data:
        # 此处能够根据需要调整对应的embedding
        sentence = one_data['sentence']
        trigger = one_data['trigger']
        trigger_site = int(one_data['trigger_site'])
        embedding = score_sentence_feature(sentence, trigger, trigger_site, embeddings_index)
        event_embeddings.append(copy.deepcopy(embedding))
    # 加载聚类的结果
    with open(cluster_res_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        cluster_res = json.loads(json_str)
    cluster_data = defaultdict(list)
    for one_data in cluster_res:
        cluster_data[one_data['cluster']].append(one_data)

    # 分别计算每一类的评分
    cluster_score = defaultdict(float)
    for key in cluster_data.keys():
        buf = cluster_data[key]
        one_cluster_score = 0
        num_of_inf = 0
        for one_buf in buf:
            dot_product_max = float("-inf")
            embedding = score_sentence_feature(one_buf['sentence'], one_buf['trigger'],
                                               one_buf['trigger_site'], embeddings_index)
            for event_embedding in event_embeddings:
                sim_res = similarity(embedding, event_embedding)
                if sim_res is not None:
                    if dot_product_max < sim_res:
                        dot_product_max = sim_res
            if dot_product_max != float("-inf"):
                one_cluster_score += dot_product_max
            else:
                num_of_inf += 1
        cluster_score[key] = one_cluster_score/(len(buf)-num_of_inf)
    print(str(cluster_score))

    '''
    # 返回评分最大的一类
    max_score = float("-inf")
    positive_cluster = None
    for key in cluster_score:
        if max_score < cluster_score[key]:
            max_score = cluster_score[key]
            positive_cluster = key
    positive_data = cluster_data[positive_cluster]

    # 所其他的类拼接起来作为负例
    negative_data = []
    for key in cluster_data.keys():
        if key != positive_cluster:
            negative_data = negative_data + cluster_data[key]
    '''
    threshold = 0.9
    positive_data = []
    negative_data = []
    # 将评分大于阈值的作为的正例，小于阈值的作为负例
    for key in cluster_score:
        if threshold < cluster_score[key]:
            positive_data = positive_data + cluster_data[key]
        else:
            negative_data = negative_data + cluster_data[key]

    # 分别返回正例与负例
    print("聚类结果中正例数量：", len(positive_data))
    print("聚类结果中负例数量：", len(negative_data))
    return positive_data, negative_data


# 对聚类的结果进行评估
def cluster_result_evaluation(cluster_res_file, embedding_index):
    # 读取聚类结果，并按照类别生成列表
    with open(cluster_res_file, 'r') as f:
        json_str = '['
        for line in f.readlines():
            json_str = json_str + line.strip('\n') + ','
        json_str = json_str.strip(',') + ']'
        json_str = json_str.replace("'", '"')
        cluster_res = json.loads(json_str)
    cluster_data = defaultdict(list)
    for one_data in cluster_res:
        cluster_data[one_data['cluster']].append(one_data)

    # 针对每一个句子进行评估
    for one_cluster in cluster_data:
        other_clusters = []
        for other_cluster in cluster_data:
            if other_cluster != one_cluster:
                other_clusters += other_cluster
        for one_data in one_cluster:
            ICD = calculate_inter_cluster_dissimilarity(one_data, one_cluster, embedding_index)
            OCD = calculate_outer_cluster_dissimilarity(one_data, other_clusters, embedding_index)
            print(one_data['sentence'], ICD, OCD)
            break
        break


# 计算某一个句子的簇内相似度
def calculate_inter_cluster_dissimilarity(this_data, inter_cluster, embedding_index):
    this_data_feature = cluster_sentence_feature(this_data['sentence'], this_data['trigger'], this_data['trigger_site'], embedding_index)
    sum_distance = 0
    for one_data in inter_cluster:
        if one_data == this_data:
            continue
        one_data_feature = cluster_sentence_feature(one_data['sentence'], one_data['trigger'], one_data['trigger_site'], embedding_index)
        one_distance = distance_trigger(this_data_feature, one_data_feature)
        sum_distance += one_distance
    return sum_distance/(len(inter_cluster) - 1)


# 计算某一个句子的簇外相似度
def calculate_outer_cluster_dissimilarity(this_data, outer_clusters, embedding_index):
    this_data_feature = cluster_sentence_feature(this_data['sentence'], this_data['trigger'], this_data['trigger_site'], embedding_index)
    res = float("inf")
    for outer_cluster in outer_clusters:
        sum_distance = 0
        for one_data in outer_cluster:
            one_data_feature = cluster_sentence_feature(one_data['sentence'], one_data['trigger'], one_data['trigger_site'], embedding_index)
            one_distance = distance_trigger(this_data_feature, one_data_feature)
            sum_distance += one_distance
        avg_distance = sum_distance/len(outer_cluster)
        if avg_distance < res:
            res = avg_distance
    return res


# 计算某一个句子的轮廓系数
def calculate_silhouette_coefficient(inter, outer):
    return (outer - inter)/max(inter, outer)


def k_means_cluster(alignment_file, cluster_res):
    # 获取聚类数据，第一个为特征，第二个为数据，后者为对应句子
    embedding_index, embedding_length = dataProcess.get_chinese_embedding()
    k_means_input, input_data, sentences = get_input(alignment_file, embedding_index)
    class_result = k_means(k_means_input, 4)
    file = open(cluster_res, 'w')
    for index, item in enumerate(class_result):
        input_data[index]['sentence'] = sentences[index]
        input_data[index]['cluster'] = item
        file.write(str(input_data[index]) + '\n')
    file.close()
    # show_result("第一轮迭代/cluster_res_team_and_score.txt", 4)
    # score_to_cluster("第一轮迭代/cluster_res_team_and_score.txt")

    # score_to_cluster("第一轮迭代/cluster_res_only_team.txt")
    # show_result("第一轮迭代/cluster_res_only_team.txt", 4)


if __name__ == '__main__':
    embedding_index, embedding_length = dataProcess.get_chinese_embedding()
    print("词向量加载完毕！")
    cluster_result_evaluation("第三轮迭代/cluster_res.txt", embedding_index)
