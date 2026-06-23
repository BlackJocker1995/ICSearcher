import json
import os

import numpy as np
import pandas as pd
from pymavlink import mavextra
from Cptool.config import toolConfig

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
    path = toolConfig._param_file(toolConfig.MODE)
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))


def load_sub_param():
    """
    load parameter we want to fuzzing
    :return:
    """
    path = toolConfig._param_file(toolConfig.MODE)
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))[toolConfig.PARAM_PART]


def get_default_values(para_dict):
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    return para_dict.loc['step'].to_numpy()


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
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def return_min_max_scaler(trans, values):
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = return_min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def pad_configuration_default_value(params_value):
    para_dict = load_param()
    # default values
    all_default_value = para_dict.loc[['default']]
    all_default_value = pd.concat([all_default_value]*params_value.shape[0])
    # replace values
    participle_param = toolConfig.PARAM_PART
    all_default_value[participle_param] = params_value
    return all_default_value.values