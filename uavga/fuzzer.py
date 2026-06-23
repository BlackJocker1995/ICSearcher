import colorsys
import pickle
import random
from abc import abstractmethod
import scipy.cluster.hierarchy as hcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN
from sklearn.decomposition import PCA
from loguru import logger

from Cptool.config import toolConfig
from Cptool.gaSimManager import GaSimManager
from Cptool.mavtool import min_max_scaler_param
from ModelFit.approximate import CyLSTM, Modeling
from uavga.problem import ProblemGA
from uavga.searcher import GAOptimizer


def split_segment(csv_data):
    """
    select status data and split to multiple segments
    :return:
    """
    # Drop configuration (parameter values)
    tmp = csv_data.to_numpy()[:, :toolConfig.STATUS_LEN]
    # To prevent unbalanced
    tmp = tmp[:-(tmp.shape[0] % (toolConfig.SEGMENT_LEN + 1)), :]
    # Split
    tmp_split = np.array_split(tmp, tmp.shape[0] // (toolConfig.SEGMENT_LEN + 1), axis=0)

    return np.array(tmp_split)


def random_choice_dbscan(segment_csv, eps=0.5):
    # 3D to 2D
    data_class = segment_csv.reshape(
        (-1, segment_csv.shape[1] * segment_csv.shape[2]))
    # Cluster
    clf = DBSCAN(eps=eps, min_samples=5).fit(data_class)
    # Cluster reuslt
    predicted = clf.labels_
    logger.info(f'DBSCAN class: {max(predicted)}')

    # ------------- draw ------------------#
    c = list(map(lambda x: color(tuple(x)), ncolors(max(predicted) + 1)))

    colors = [c[i] for i in predicted]

    pca = PCA(n_components=2, svd_solver='arpack')
    show = pca.fit_transform(data_class)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=colors, s=5)
    plt.scatter(show[:, 0], show[:, 1], c=colors, s=5)

    plt.show()
    # -------------------------

    out = []
    for i in range(max(predicted)):
        index = np.where(predicted == i)[0]
        col_index = np.random.choice(index, min(index.shape[0], toolConfig.CLUSTER_CHOICE_NUM))
        select = segment_csv[col_index]
        out.extend(select)
    out = np.array(out)
    return out


def random_choice_hierarchical(segment_csv, rate=0.5):
    # 3D to 2D
    data_class = segment_csv.reshape(
        (-1, segment_csv.shape[1] * segment_csv.shape[2]))

    # clustering
    thresh = 0.3
    predicted = hcluster.fclusterdata(data_class, thresh, criterion="distance")
    # Cluster reuslt
    logger.info(f'Meanshift class: {max(predicted)}')
    # ------------- draw ------------------#
    for i in range(max(predicted)):
        index = np.where(predicted == i)[0]

    c = list(map(lambda x: color(tuple(x)), ncolors(max(predicted) + 1)))

    colors = [c[i] for i in predicted]

    pca = PCA(n_components=2, svd_solver='arpack')
    show = pca.fit_transform(data_class)

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=colors, s=5)
    plt.scatter(show[:, 0], show[:, 1], c=colors, s=5)

    plt.show()
    # -------------------------

    out = []
    for i in range(max(predicted)):
        index = np.where(predicted == i)[0]
        col_index = np.random.choice(index, min(index.shape[0], toolConfig.CLUSTER_CHOICE_NUM))
        select = segment_csv[col_index]
        out.extend(select)
    out = np.array(out)
    return out


def run_fuzzing(np_data, num=0):
    """
    Start Fuzzing
    :param num: The number of data to join cluster
    :return:
    """

    predictor = CyLSTM(100, 100, toolConfig.DEBUG)
    predictor.read_model()

    gaOptimizer = GAOptimizer()
    gaOptimizer.set_bounds()
    gaOptimizer.set_predictor(predictor)

    segment_csv = np_data
    # meanshift cluster
    if num != 0:
        # Random select
        index = np.random.choice(np.arange(segment_csv.shape[0]), num)
        segment_csv = segment_csv[index, :, :]
    segment_csv = random_choice_dbscan(segment_csv, eps=0.4)

    obj_population = []  # 种群

    for i, context in enumerate(segment_csv):
        # Pre process
        context = np.c_[context, np.zeros((context.shape[0], len(toolConfig.PARAM)))]
        context = Modeling.series_to_supervised(context, toolConfig.INPUT_LEN, toolConfig.OUTPUT_LEN).values

        gaOptimizer.problem.init_status(context)
        gaOptimizer.start_optimize()
        obj_population.append(gaOptimizer.population)

        print(f'------------------- {i + 1} / {segment_csv.shape[0]} -----------------')
    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'wb') as f:
        pickle.dump(obj_population, f)


def return_best_n_gen(n=1):
    candidate_vars = []
    candidate_objs = []

    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'rb') as f:
        obj_populations = pickle.load(f)
    for pop in obj_populations:
        pop_v = pop.ObjV
        pop_p = pop.Phen

        candidate_var_index = np.unique(pop_p, axis=0, return_index=True)[1]

        pop_v = pop_v[candidate_var_index]
        pop_p = pop_p[candidate_var_index]

        candidate = [-1] * pop_v

        candidate_index = np.argsort(candidate.reshape(-1))
        pop_v = pop_v[candidate_index].reshape((-1, 1))
        pop_p = pop_p[candidate_index].reshape((-1, 20))

        if n != 0:
            candidate_var = pop_v[:min(n, len(pop_v))]
            candidate_obj = pop_p[:min(n, len(pop_p))]
        candidate_obj = ProblemGA.reasonable_range_static(candidate_obj)

        candidate_vars.extend(candidate_var)
        candidate_objs.extend(candidate_obj)

    return candidate_vars, candidate_objs


def return_random_n_gen(n=1):
    candidate_vars = []
    candidate_objs = []

    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'rb') as f:
        obj_populations = pickle.load(f)
    for pop in obj_populations:
        pop_v = pop.ObjV
        pop_p = pop.Phen
        # Unique
        candidate_var_index = np.unique(pop_p, axis=0, return_index=True)[1]
        # Choice
        pop_v = pop_v[candidate_var_index]
        pop_p = pop_p[candidate_var_index]
        #
        candidate = [-1] * pop_v
        # Sort
        candidate_index = np.argsort(candidate.reshape(-1))
        pop_v = pop_v[candidate_index].reshape((-1, 1))
        pop_p = pop_p[candidate_index].reshape((-1, 20))
        #
        if n != 0:
            candidate_var = pop_v[[1, 200, 500]]
            candidate_obj = pop_p[[1, 200, 500], :]
        candidate_obj = ProblemGA.reasonable_range_static(candidate_obj)

        candidate_vars.extend(candidate_var)
        candidate_objs.extend(candidate_obj)
    return candidate_vars, candidate_objs


def return_cluster_thres_gen(thres=0.4):
    candidate_vars = []
    candidate_objs = []

    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'rb') as f:
        obj_populations = pickle.load(f)
    for pop in obj_populations:
        pop_v = pop.ObjV
        pop_p = pop.Phen
        # Unique
        candidate_var_index = np.unique(pop_p, axis=0, return_index=True)[1]
        # Choice
        pop_v = pop_v[candidate_var_index]
        pop_p = pop_p[candidate_var_index]

        # To normal value
        normal_pop_p = ProblemGA.reasonable_range_static(pop_p)
        normalize_pop_p = min_max_scaler_param(normal_pop_p)

        predicted = hcluster.fclusterdata(normalize_pop_p, thres, criterion="distance")
        # ------------- draw ------------------#
        # c = list(map(lambda x: color(tuple(x)), ncolors(max(predicted) + 1)))
        #
        # colors = [c[i] for i in predicted]
        #
        # pca = PCA(n_components=2, svd_solver='arpack')
        # show = pca.fit_transform(pop_p)
        #
        # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # ax.scatter(show[:, 0], show[:, 1], show[:, 2], c=colors, s=5)
        # plt.scatter(show[:, 0], show[:, 1], c=colors, s=5)
        #
        # plt.show()
        # -------------------------
        for i in range(max(predicted)):
            index = np.where(predicted == i)[0]
            col_index = np.random.choice(index, min(index.shape[0], toolConfig.CLUSTER_CHOICE_NUM))
            if len(col_index) > 0:
                candidate_vars.extend(pop_v[col_index])
                candidate_objs.extend(normal_pop_p[col_index])
    # Array
    candidate_objs = np.array(candidate_objs).astype(float)
    candidate_vars = np.array(candidate_vars).astype(float)
    return candidate_vars, candidate_objs


def reshow(params, values):
    manager = GaSimManager(debug=toolConfig.DEBUG)
    manager.start_multiple_sitl()
    manager.mav_monitor_init()

    manager.mav_monitor_connect()
    manager.mav_monitor_set_mission(toolConfig.resolve("Cptool/mission.txt"), random=True)

    manager.mav_monitor_set_param(params=params, values=values)

    # manager.start_mav_monitor()
    manager.mav_monitor_start_mission()
    result = manager.mav_monitor_error()

    manager.stop_sitl()


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors


def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def color(value):
    digit = list(map(str, range(10))) + list("ABCDEF")
    if isinstance(value, tuple):
        string = '#'
        for i in value:
            a1 = i // 16
            a2 = i % 16
            string += digit[a1] + digit[a2]
        return string
    elif isinstance(value, str):
        a1 = digit.index(value[1]) * 16 + digit.index(value[2])
        a2 = digit.index(value[3]) * 16 + digit.index(value[4])
        a3 = digit.index(value[5]) * 16 + digit.index(value[6])
        return (a1, a2, a3)
