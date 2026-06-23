import argparse
import csv
from loguru import logger
import os
import pickle
import time

import numpy as np
import pandas as pd

import Cptool
import ModelFit
from Cptool.config import toolConfig
from Cptool.gaMavlink import GaMavlinkAPM
from Cptool.gaSimManager import GaSimManager


# from Cptool.gaSimManager import GaSimManager
from uavga.fuzzer import return_random_n_gen, return_cluster_thres_gen

if __name__ == '__main__':
    toolConfig.select_mode("PX4")
    # Get Fuzzing result and validate
    candidate_var, candidate_obj = return_cluster_thres_gen(0.35)
    candidate_obj = np.array(candidate_obj, dtype=float).round(8)
    candidate_var = np.array(candidate_var, dtype=float).round(8)

    with open(f'result/{toolConfig.MODE}/pop{toolConfig.EXE}.pkl', 'wb') as f:
        pickle.dump([candidate_obj, candidate_var], f)

