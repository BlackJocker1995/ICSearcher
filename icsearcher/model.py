# -*- coding:utf-8 -*-
"""Surrogate model: LSTM / TCN predictors for flight-status deviation (PyTorch).

Stage-4 rewrite of the legacy Keras ``Modeling``/``CyLSTM``/``CyTCN``. The
time-series data utilities (series_to_supervised, fit_trans/load_trans,
merge_file_data, cal_patch_deviation, data_split*) are preserved verbatim;
only the model build/train/persist/predict path moved from Keras to PyTorch.

The on-disk artifact changed from ``lstm.h5`` to ``lstm.pt`` (a state-dict
dump), so previously trained Keras models are not loadable — retrain via
``pipelines/2_train.py train`` after this stage.
"""
import os
import pickle
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from icsearcher import config as _config_mod
from icsearcher.params import min_max_scaler


def _config():
    """Lazy config access so per-mode monkeypatching of the singleton works.

    Importing ``toolConfig`` at module top-level binds the singleton object
    captured at first import, which defeats per-mode monkeypatching (the same
    reason ``icsearcher.params`` reads it through a function). Reading it
    through the module at call time picks up any reassignment of
    ``icsearcher.config.toolConfig``.
    """
    return _config_mod.toolConfig


class _LSTMNet(nn.Module):
    """LSTM(128) -> Dropout -> Dense(64) -> Dense(64) -> Dense(output).

    Mirrors the legacy Keras Sequential topology.
    """

    def __init__(self, n_features: int, input_len: int, output_len: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=128, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_len),
        )

    def forward(self, x):
        out, _ = self.lstm(x)        # (batch, seq, 128)
        out = out[:, -1, :]          # take the last timestep -> (batch, 128)
        return self.fc(out)


class Modeling(object):
    """Base class for time-series modeling and prediction.

    Subclasses implement the PyTorch ``nn.Module`` factory and the
    sample/label split. Train/predict/persist live here and are framework-aware.
    """

    def __init__(self, debug: bool = False):
        self._model: nn.Module = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_out = f"{_config().INPUT_LEN}_{_config().OUTPUT_LEN}"

    # ------------------------------------------------------------------ data utils
    @classmethod
    def cs_to_sl(cls, values):
        """Convert a continuous series to supervised-learning form."""
        cfg = _config()
        values = values.astype('float32')
        if cfg.RETRANS:
            trans = cls.load_trans()
            values = min_max_scaler(trans, values)
        reframed = cls.series_to_supervised(values, cfg.INPUT_LEN,
                                            cfg.OUTPUT_LEN, True)
        model_dir = f'model/{cfg.MODE}/{cfg.INPUT_LEN}_{cfg.OUTPUT_LEN}'
        os.makedirs(model_dir, exist_ok=True)
        return reframed

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def _train_valid_split(self, values):
        X, Y = self.data_split(values)
        train_X, valid_X, train_Y, valid_Y = train_test_split(X, Y, test_size=0.2, random_state=2022)
        logger.info(f"Shape: {train_X.shape}, {train_Y.shape}, {valid_X.shape}, {valid_Y.shape}")
        return train_X, train_Y, valid_X, valid_Y

    def _test_split(self, values):
        X, Y = self.data_split(values)
        logger.info(f"Shape: {X.shape}, {Y.shape}")
        return X, Y

    def read_trans(self):
        self.trans = self.load_trans()

    def extract_feature(self, dir):
        file_list = sorted(f for f in os.listdir(dir) if f.endswith(".csv"))
        pd_array = None
        for filename in file_list:
            data = pd.read_csv(f"{dir}/{filename}").drop(["TimeS"], axis=1)
            values = self.cs_to_sl(data.values)
            pd_array = values if pd_array is None else pd.concat([pd_array, values])
        return pd_array

    @abstractmethod
    def data_split(self, value):
        pass

    @abstractmethod
    def _build_module(self, n_features: int) -> nn.Module:
        """Return the PyTorch nn.Module for this surrogate."""
        pass

    @abstractmethod
    def _artifact_name(self) -> str:
        """On-disk artifact name, e.g. 'lstm.pt'."""
        pass

    # ------------------------------------------------------------------ train/predict
    def _model_path(self):
        return f'model/{_config().MODE}/{self.in_out}/{self._artifact_name()}'

    def _ensure_module(self, n_features):
        if self._model is None:
            self._model = self._build_module(n_features).to(self._device)

    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        if not getattr(self, 'epochs', None):
            raise ValueError('set model param (epochs) at first!')
        train_X = np.asarray(train_X, dtype='float32')
        n_features = train_X.shape[2]
        self._ensure_module(n_features)
        model = self._model
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        def _tensor(arr):
            return torch.from_numpy(np.asarray(arr, dtype='float32')).to(self._device)

        X_t, Y_t = _tensor(train_X), _tensor(train_Y)
        X_v, Y_v = _tensor(valid_X), _tensor(valid_Y)

        losses, val_losses = [], []
        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(X_t)
            loss = criterion(pred, Y_t)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

            model.eval()
            with torch.no_grad():
                v_pred = model(X_v)
                v_loss = criterion(v_pred, Y_v)
            val_losses.append(float(v_loss.item()))

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                logger.info(f"epoch {epoch}: loss={loss.item():.6f} val_loss={v_loss.item():.6f}")

        suffix = str(num) if num is not None else ""
        path = self._model_path().replace('.pt', f'{suffix}.pt')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        self._plot_loss(losses, val_losses, num)
        return model

    def _plot_loss(self, losses, val_losses, num):
        import matplotlib.pyplot as plt
        label = f'train-{num}' if num is not None else 'train'
        vlabel = f'validation-{num}' if num is not None else 'validation'
        plt.figure()
        plt.plot(losses, label=label)
        plt.plot(val_losses, label=vlabel)
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs Time', fontsize=18)
        plt.legend(prop={'size': '18'})
        plt.savefig(self._model_path().replace('.pt', '_loss.pdf'))
        plt.close()

    def train(self, values, cuda: bool = False):
        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            self._device = torch.device("cpu")
        train_X, train_y, valid_X, valid_y = self._train_valid_split(values)
        self._model = self._fit_network(train_X, train_y, valid_X, valid_y)

    def _predict_numpy(self, X):
        if self._model is None:
            raise ValueError('Train or load model at first')
        X = np.asarray(X, dtype='float32')
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X).to(self._device))
        return pred.cpu().numpy()

    def predict(self, values):
        predict_X = self._predict_numpy(values)
        if _config().RETRANS:
            trans = self.load_trans()
            predict_X = trans.inverse_transform(predict_X)
        return predict_X

    def predict_feature(self, feature_data):
        """Predict on already-pre-processed feature data."""
        return self._predict_numpy(feature_data)

    def read_model(self):
        path = self._model_path()
        # Infer the feature width from the model definition lazily: build then
        # load the state dict. The module's input size is fixed at build time.
        self._model = self._build_module(self._n_features()).to(self._device)
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.eval()

    def _n_features(self) -> int:
        """Number of input features per timestep (DATA_LEN)."""
        return _config().DATA_LEN

    def status2feature(self, status_data):
        if "TimeS" in status_data.columns:
            status_data = status_data.drop(["TimeS"], axis=1)
        return self.cs_to_sl(status_data.values)

    # ------------------------------------------------------------------ deviation
    @classmethod
    def cal_patch_deviation(cls, predicted_data, status_data):
        deviation = np.abs(status_data - predicted_data)
        if len(predicted_data.shape) == 3:
            loss = deviation.sum(axis=tuple(range(1, 3)))
        else:
            loss = deviation.sum(axis=1).sum(axis=1)
        return loss

    cal_deviation_old = cal_patch_deviation  # back-compat alias

    def feature_deviation(self, test_data, cuda: bool = False):
        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        test_X, test_Y = self.data_split(test_data)
        pred_y = self._predict_numpy(test_X)
        deviation = np.abs(pred_y - test_Y)
        split_loss = [it.sum() for it in np.split(deviation, 1000)]
        return np.array(split_loss)

    feature_deviation_old = feature_deviation  # back-compat alias

    # ------------------------------------------------------------------ scaler
    @staticmethod
    def fit_trans(pd_csv):
        cfg = _config()
        values = pd_csv.values
        status_value = values[:, :cfg.STATUS_LEN]
        trans = MinMaxScaler(feature_range=(0, 1))
        trans.fit(status_value)
        os.makedirs(f"model/{cfg.MODE}", exist_ok=True)
        with open(f'model/{cfg.MODE}/trans.pkl', 'wb') as f:
            pickle.dump(trans, f)

    @staticmethod
    def load_trans():
        with open(f'model/{_config().MODE}/trans.pkl', 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def series2segment_predict(data, has_param=False, dropnan=True):
        cfg = _config()
        df = pd.DataFrame(data)
        cols = list()
        for i in range(cfg.INPUT_LEN - 1, -1, -1):
            cols.append(df.shift(i))
        agg = pd.concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)
        return agg.to_numpy().reshape((-1, cfg.INPUT_LEN, cfg.DATA_LEN))


class CyLSTM(Modeling):
    """LSTM surrogate predictor (PyTorch)."""

    def __init__(self, epochs: int, batch_size: int, debug: bool = False):
        super().__init__(debug)
        self.epochs = epochs
        self.batch_size = batch_size  # kept for API compat; full-batch GD is used

    def _build_module(self, n_features: int) -> nn.Module:
        cfg = _config()
        return _LSTMNet(n_features=n_features,
                        input_len=cfg.INPUT_LEN,
                        output_len=cfg.OUTPUT_DATA_LEN)

    def _artifact_name(self) -> str:
        return 'lstm.pt'

    def data_split(self, values):
        cfg = _config()
        if isinstance(values, pd.DataFrame):
            values = values.values
        X = values[:, :cfg.INPUT_DATA_LEN]
        y = values[:, cfg.INPUT_DATA_LEN:]
        y = y.reshape((y.shape[0], cfg.OUTPUT_LEN, -1))
        Y = y[:, :, :-cfg.PARAM_LEN].reshape((y.shape[0], cfg.OUTPUT_DATA_LEN))
        X = X.reshape((X.shape[0], cfg.INPUT_LEN, cfg.DATA_LEN))
        Y = Y.reshape((Y.shape[0], cfg.OUTPUT_DATA_LEN))
        return X, Y

    def data_split_3d(self, values):
        cfg = _config()
        values = values.values if isinstance(values, pd.DataFrame) else values
        X = values[:, :, :cfg.INPUT_DATA_LEN]
        y = values[:, :, cfg.INPUT_DATA_LEN:-cfg.PARAM_LEN]
        X = X.reshape((-1, cfg.INPUT_LEN, cfg.DATA_LEN))
        Y = y.reshape((-1, cfg.OUTPUT_DATA_LEN))
        return X, Y

    @classmethod
    def merge_file_data(cls, dir):
        file_list = sorted(f for f in os.listdir(dir) if f.endswith(".csv"))
        col_name = pd.read_csv(f"{dir}/{file_list[0]}").columns
        pd_csv = pd.DataFrame(columns=col_name)
        for filename in tqdm(file_list):
            pd_csv = pd.concat([pd_csv, pd.read_csv(f"{dir}/{filename}")])
        return pd_csv.drop(["TimeS"], axis=1)


class CyTCN(Modeling):
    """TCN surrogate predictor (PyTorch).

    A lightweight temporal-convolutional head replaces the keras-tcn dependency.
    Falls back to the same train/predict path; only the module differs.
    """

    def __init__(self, epochs: int, batch_size: int, debug: bool = False):
        super().__init__(debug)
        self.epochs = epochs
        self.batch_size = batch_size

    def _build_module(self, n_features: int) -> nn.Module:
        output_len = _config().OUTPUT_DATA_LEN

        class _TCNNet(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.conv = nn.Sequential(
                    nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(),
                )
                self_inner.fc = nn.Linear(64, output_len)

            def forward(self, x):
                # x: (batch, seq, features) -> Conv1d wants (batch, features, seq)
                out = self_inner.conv(x.transpose(1, 2))
                out = out[:, :, -1]   # last timestep
                return self_inner.fc(out)

        return _TCNNet()

    def _artifact_name(self) -> str:
        return 'tcn.pt'

    def data_split(self, value):
        cfg = _config()
        values = value.values
        X = values[:, :cfg.INPUT_DATA_LEN]
        y = values[:, cfg.INPUT_DATA_LEN:]
        y = y.reshape((y.shape[0], cfg.OUTPUT_LEN, -1))
        Y = y[:, :, :-cfg.PARAM_LEN].reshape((y.shape[0], cfg.OUTPUT_DATA_LEN))
        X = X.reshape((X.shape[0], cfg.INPUT_LEN, cfg.DATA_LEN))
        Y = Y.reshape((Y.shape[0], 1, cfg.OUTPUT_DATA_LEN))
        return X, Y
