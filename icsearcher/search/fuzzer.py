import colorsys
import os
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

from icsearcher.config import toolConfig
from icsearcher.sim import GaSimManager
from icsearcher.params import min_max_scaler_param
from icsearcher.model import CyLSTM, Modeling
from icsearcher.search.problem import ProblemGA
from icsearcher.search.searcher import GAOptimizer


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
    # Cluster result
    predicted = clf.labels_
    n_clusters = max(predicted) + 1
    logger.info(f'DBSCAN class: {n_clusters}')

    out = []
    for i in range(n_clusters):
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
    # Cluster result
    n_clusters = max(predicted)
    logger.info(f'Hierarchical class: {n_clusters}')

    out = []
    for i in range(n_clusters):
        index = np.where(predicted == i + 1)[0]
        col_index = np.random.choice(index, min(index.shape[0], toolConfig.CLUSTER_CHOICE_NUM))
        select = segment_csv[col_index]
        out.extend(select)
    out = np.array(out)
    return out


def run_fuzzing(np_data, num=0):
    """Start surrogate-guided fuzzing over flight-context segments.

    Args:
        np_data: held-out raw test segments (from stage 2 raw_split).
        num: if non-zero, randomly subsample this many segments before clustering.
    """
    predictor = CyLSTM(100, 100, toolConfig.DEBUG)
    predictor.read_model()

    gaOptimizer = GAOptimizer()
    gaOptimizer.set_predictor(predictor)

    segment_csv = np_data
    if num != 0:
        index = np.random.choice(np.arange(segment_csv.shape[0]), num)
        segment_csv = segment_csv[index, :, :]
    segment_csv = random_choice_dbscan(segment_csv, eps=0.4)

    obj_population = []  # one PopulationResult per flight context

    for i, context in enumerate(segment_csv):
        # Pre-process: append a zeroed param block and reframe to supervised form.
        context = np.c_[context, np.zeros((context.shape[0], len(toolConfig.PARAM)))]
        context = Modeling.series_to_supervised(context, toolConfig.INPUT_LEN, toolConfig.OUTPUT_LEN).values

        gaOptimizer.problem.init_status(context)
        gaOptimizer.start_optimize()
        obj_population.append(gaOptimizer.population)

        print(f'------------------- {i + 1} / {segment_csv.shape[0]} -----------------')
    out_dir = f'result/{toolConfig.MODE}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/pop{toolConfig.EXE}.pkl', 'wb') as f:
        pickle.dump(obj_population, f)


def _load_populations():
    """Load the per-context populations written by run_fuzzing."""
    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'rb') as f:
        return pickle.load(f)


def return_best_n_gen(n=1):
    """Return the n best (var, obj) candidates across all contexts.

    ``ObjV`` holds the negated deviation (minimize), so ascending order is
    best-first. Returns (candidate_vars, candidate_objs) where objs are the
    step-scaled original-unit params.
    """
    candidate_vars, candidate_objs = [], []
    for pop in _load_populations():
        pop_v, pop_p = pop.ObjV, pop.Phen
        unique = np.unique(pop_p, axis=0, return_index=True)[1]
        pop_v, pop_p = pop_v[unique], pop_p[unique]
        order = np.argsort(pop_v.reshape(-1))
        pop_v, pop_p = pop_v[order], pop_p[order]
        k = min(n, len(pop_v)) if n else len(pop_v)
        candidate_vars.extend(pop_v[:k])
        candidate_objs.extend(ProblemGA.reasonable_range_static(pop_p[:k]))
    return np.array(candidate_vars).astype(float), np.array(candidate_objs).astype(float)


def return_random_n_gen(n=3):
    """Return a few randomly-positioned candidates per context."""
    candidate_vars, candidate_objs = [], []
    for pop in _load_populations():
        pop_v, pop_p = pop.ObjV, pop.Phen
        unique = np.unique(pop_p, axis=0, return_index=True)[1]
        pop_v, pop_p = pop_v[unique], pop_p[unique]
        m = len(pop_v)
        if m == 0:
            continue
        picks = np.random.choice(np.arange(m), size=min(n, m), replace=False)
        candidate_vars.extend(pop_v[picks])
        candidate_objs.extend(ProblemGA.reasonable_range_static(pop_p[picks]))
    return np.array(candidate_vars).astype(float), np.array(candidate_objs).astype(float)


def return_cluster_thres_gen(thres=0.4):
    """Cluster each context's population and sample diverse candidates.

    This is the selector used by ``pipelines/4_validate.py pre``. Each context's
    step-scaled, normalized population is hierarchically clustered at ``thres``;
    up to ``CLUSTER_CHOICE_NUM`` candidates per cluster are kept.
    """
    candidate_vars, candidate_objs = [], []
    for pop in _load_populations():
        pop_v, pop_p = pop.ObjV, pop.Phen
        unique = np.unique(pop_p, axis=0, return_index=True)[1]
        pop_v, pop_p = pop_v[unique], pop_p[unique]

        normal_pop_p = ProblemGA.reasonable_range_static(pop_p)
        normalize_pop_p = min_max_scaler_param(normal_pop_p)

        predicted = hcluster.fclusterdata(normalize_pop_p, thres, criterion="distance")
        for i in range(max(predicted)):
            index = np.where(predicted == i)[0]
            col_index = np.random.choice(index, min(index.shape[0], toolConfig.CLUSTER_CHOICE_NUM))
            if len(col_index) > 0:
                candidate_vars.extend(pop_v[col_index])
                candidate_objs.extend(normal_pop_p[col_index])
    candidate_objs = np.array(candidate_objs).astype(float)
    candidate_vars = np.array(candidate_vars).astype(float)
    return candidate_vars, candidate_objs


def reshow(params, values):
    manager = GaSimManager(debug=toolConfig.DEBUG)
    manager.start_multiple_sitl()
    manager.mav_monitor_init()

    manager.mav_monitor_connect()
    manager.mav_monitor_set_mission(toolConfig.resolve("data/mission.txt"), random=True)

    manager.mav_monitor_set_param(params=params, values=values)

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
