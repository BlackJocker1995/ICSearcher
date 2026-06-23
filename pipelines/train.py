"""Stage 2 — feature engineering, train/test split, and LSTM training.

Subsumes the old 2.extract_feature / 2.feature_split / 2.raw_split / 2.train_lstm
scripts (and their _px4 twins). The firmware is chosen by ``data/config.yaml``'s
``mode`` field.

Usage:
    python pipelines/2_train.py extract     # build features + fit scaler
    python pipelines/2_train.py split       # split features into train/test
    python pipelines/2_train.py raw_split   # carve held-out raw test segments
    python pipelines/2_train.py train       # train the LSTM
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from icsearcher.config import toolConfig
from icsearcher.model import CyLSTM, Modeling


def _feature_dir():
    return f"model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}"


def _converted_csv_dir():
    """Directory of converted flight-log CSVs, per firmware."""
    if toolConfig.MODE == "PX4":
        return f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/ulg_changed/csv"
    return f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/bin_regular/csv"


def extract():
    pd_csv = CyLSTM.merge_file_data(_converted_csv_dir())
    CyLSTM.fit_trans(pd_csv)

    lstm = CyLSTM(100, 512)
    feature = lstm.extract_feature(_converted_csv_dir())
    feature.to_csv(f"{_feature_dir()}/features.csv", index=False)


def split():
    feature = pd.read_csv(f"{_feature_dir()}/features.csv")
    index = np.arange(0, feature.shape[0])
    train, test = train_test_split(index, test_size=0.1, shuffle=True, random_state=2022)
    feature.iloc[train].to_csv(f"{_feature_dir()}/features_train.csv", index=False)
    feature.iloc[test].to_csv(f"{_feature_dir()}/features_test.csv", index=False)


def raw_split():
    pd_csv = CyLSTM.merge_file_data(_converted_csv_dir())
    np_data = pd_csv.to_numpy()[:, :toolConfig.STATUS_LEN]

    trans = Modeling.load_trans()
    np_data = trans.transform(np_data)

    # Drop the partial tail so the data splits evenly into segments.
    np_data = np_data[:-(np_data.shape[0] % (toolConfig.SEGMENT_LEN + 1)), :]
    np_data = np.array(np.array_split(np_data, np_data.shape[0] // (toolConfig.SEGMENT_LEN + 1), axis=0))

    index = np.arange(0, np_data.shape[0])
    _, test = train_test_split(index, test_size=0.1, shuffle=True, random_state=2022)

    with open(f"model/{toolConfig.MODE}/raw_test.pkl", "wb") as f:
        pickle.dump(np_data[test], f)


def train():
    lstm = CyLSTM(100, 512)
    feature = pd.read_csv(f"{_feature_dir()}/features_train.csv")
    lstm.train(feature, cuda=True)


def main():
    parser = argparse.ArgumentParser(description="Stage 2: feature engineering + training")
    parser.add_argument("step", choices=["extract", "split", "raw_split", "train"],
                        help="which sub-step to run (run in this order)")
    args = parser.parse_args()
    {"extract": extract, "split": split, "raw_split": raw_split, "train": train}[args.step]()


if __name__ == '__main__':
    main()
