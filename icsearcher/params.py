import json
import os

import numpy as np
import pandas as pd
from pymavlink import mavextra


def _config():
    """Lazy import so tests can repoint the toolConfig singleton per mode.

    Importing at module top-level binds the singleton object captured at first
    import, which defeats per-mode monkeypatching. Reading it through the module
    at call time picks up any reassignment of ``icsearcher.config.toolConfig``.
    """
    from icsearcher.config import toolConfig
    return toolConfig


class Location:
    def __init__(self, x, y=None, timeS=0):
        if y is None:
            self.x = x.x
            self.y = x.y
        else:
            self.x = x
            self.y = y
        self.timeS = timeS
        self.npa = np.array([x, y])

    def __sub__(self, other):
        return Location(self.x-other.x, self.y-other.y)

    def __str__(self):
        return f"X: {self.x} ; Y: {self.y}"

    def sum(self):
        return self.npa.sum()

    @classmethod
    def distance(cls, point1, point2):
        return mavextra.distance_lat_lon(point1.x, point1.y,
                                         point2.x, point2.y)

def load_param():
    """
    load parameter we want to fuzzing
    :return:
    """
    path = _config()._param_file(_config().MODE)
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))


def load_sub_param():
    """
    load parameter we want to fuzzing
    :return:
    """
    cfg = _config()
    path = cfg._param_file(cfg.MODE)
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))[cfg.PARAM_PART]


def get_default_values(para_dict):
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    # Coerce to float: the param JSON mixes integer and decimal step values,
    # so pandas infers an object dtype, and ``param * step_unit`` then propagates
    # object dtype — which breaks downstream (np.isfinite, model input, etc.).
    return para_dict.loc['step'].to_numpy(dtype=float)


# Log analysis function
def read_path_specified_file(log_path, exe):
    """
        :param log_path:
        :param exe:
        :return:
        """
    file_list = []
    for filename in os.listdir(log_path):
        if filename.endswith(f'.{exe}'):
            file_list.append(filename)
    file_list.sort()
    return file_list


def rename_bin(log_path, ranges):
    file_list = read_path_specified_file(log_path, 'BIN')
    # 列出文件夹内所有.BIN结尾的文件并排序
    for file, num in zip(file_list, range(ranges[0], ranges[1])):
        name, _ = file.split('.')
        os.rename(f"{log_path}/{file}", f"{log_path}/{str(num).zfill(8)}.BIN")


def min_max_scaler_param(param_value):
    # If param.shape != predictor's all params.
    if param_value.shape[1] != load_param().shape[1]:
        para_dict = load_sub_param()
    else:
        para_dict = load_param()
    param_choice_dict = para_dict
    #participle_param = toolConfig.PARAM
    #param_choice_dict = select_sub_dict(para_dict, participle_param)

    param_bounds = read_range_from_dict(param_choice_dict)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value - lb) / (ub-lb)
    return param_value.astype(float)


def return_min_max_scaler_param(param_value: object) -> object:
    param = load_param()
    param_bounds = read_range_from_dict(param)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value * (ub-lb)) + lb
    return param_value


def min_max_scaler(trans, values):
    status_len = _config().STATUS_LEN
    status_value = values[:, :status_len]
    param_value = values[:, status_len:]

    param_value = min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def return_min_max_scaler(trans, values):
    status_len = _config().STATUS_LEN
    status_value = values[:, :status_len]
    param_value = values[:, status_len:]

    param_value = return_min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def pad_configuration_default_value(params_value):
    para_dict = load_param()
    # default values
    all_default_value = para_dict.loc[['default']]
    all_default_value = pd.concat([all_default_value]*params_value.shape[0])
    # replace values
    participle_param = _config().PARAM_PART
    all_default_value[participle_param] = params_value
    return all_default_value.values