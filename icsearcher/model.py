# -*- coding:utf-8 -*-
from loguru import logger
import os
import pickle
import time
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout, RepeatVector
from keras.layers import LSTM
from keras.models import Sequential
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model
from tqdm import tqdm

try:
    # Optional TCN backend (PyPI package: keras-tcn). Only needed if you use
    # the CyTCN surrogate; CyLSTM works without it.
    from tcn import TCN
except ImportError:  # pragma: no cover - optional dependency
    TCN = None

from icsearcher.config import toolConfig
from icsearcher.params import min_max_scaler


class Modeling(object):
    """Base class for time series modeling and prediction"""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the model
        Args:
            debug: Enable debug logging if True
        """
        self._model: Sequential = None
        self.in_out = f"{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}"

    @classmethod
    def cs_to_sl(cls, values):
        """
        Convert continuous series to supervised learning format
        Args:
            values: Input time series data
        Returns:
            Formatted data for supervised learning
        """
        values = values.astype('float32')

        # Normalize features if retransformation is enabled
        if toolConfig.RETRANS:
            trans = cls.load_trans()
            values = min_max_scaler(trans, values)

        reframed = cls.series_to_supervised(values, toolConfig.INPUT_LEN, 
                                          toolConfig.OUTPUT_LEN, True)

        # Ensure model directory exists
        model_dir = f'model/{toolConfig.MODE}/{toolConfig.INPUT_LEN}_{toolConfig.OUTPUT_LEN}'
        os.makedirs(model_dir, exist_ok=True)

        return reframed

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        convert series to supervised learning
        :param data:
        :param n_in:
        :param dropnan:
        :return:
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def _train_valid_split(self, values):
        # split into train and test sets
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
        file_list = []
        for filename in os.listdir(dir):
            if filename.endswith(".csv"):
                file_list.append(filename)
        file_list.sort()

        pd_array = None
        for index, filename in enumerate(file_list):
            # Read file
            data = pd.read_csv(f"{dir}/{filename}")
            data = data.drop(["TimeS"], axis=1)
            # extract patch
            values = data.values
            values = self.cs_to_sl(values)
            # if first
            if index == 0:
                pd_array = values
            else:
                pd_array = pd.concat([pd_array, values])
        return pd_array

    def extract_feature_separately(self, dir):
        """
        Extract the feature but not merge.
        :param dir:
        :return:
        """
        file_list = []
        # Create folder
        if not os.path.exists(dir + "/single"):
            os.makedirs(dir + "/single")
        # Read all csv
        for filename in os.listdir(dir + "/csv"):
            if filename.endswith(".csv"):
                file_list.append(filename)
        file_list.sort()

        for index, filename in enumerate(file_list):
            # Read file
            data = pd.read_csv(f"{dir}/{filename}")
            data = data.drop(["TimeS"], axis=1)
            # extract patch
            values = data.values
            values = self._cs_to_sl(values)
            # if first
            values.to_csv(f"{dir}/single/{filename}", index=False)

    @abstractmethod
    def data_split(self, value):
        pass

    @abstractmethod
    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        return None

    @abstractmethod
    def _build_model(self, train_shape: np.shape):
        return None

    @abstractmethod
    def read_model(self):
        pass

    def set_model(self, path):
        local = os.getcwd()
        self._model = load_model(f"{local}/{path}")

    def train(self, values, cuda: bool = False):
        train_X, train_y, valid_X, valid_y = self._train_valid_split(values)
        model = self._fit_network(train_X, train_y, valid_X, valid_y)
        self._model = model

    def predict(self, values):
        if self._model is None:
            logger.warning('Model is not trained!')
            raise ValueError('Train or load model at first')

        predict_X = self._model.predict(values)
        # data retrans
        if toolConfig.RETRANS:
            trans = self.load_trans()
            # trans
            predict_X = trans.inverse_transform(predict_X)

        return predict_X

    def status2feature(self, status_data):
        # extract patch
        if "TimeS" in status_data.columns:
            status_data = status_data.drop(["TimeS"], axis=1)
        values = status_data.values
        values = self.cs_to_sl(values)
        return values

    def predict_feature(self, feature_data):
        """
        predict feature which has been pre-processed
        :param feature_data:
        :return:
        """
        if self._model is None:
            logger.warning('Model is not trained!')
            raise ValueError('Train or load model at first')
        # predict each status
        predict_feature = self._model.predict(feature_data)

        return predict_feature

    def test_cmp_draw(self, test, cmp_name, num=150, exec='pdf'):
        """
        Draw comparison plots between predicted and actual values
        Args:
            test: Test data
            cmp_name: Name for comparison plots
            num: Number of samples to plot
            exec: Output file format
        """
        if self._model is None:
            raise ValueError('Train or load model first')

        # Create output directory
        fig_dir = f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}'
        os.makedirs(fig_dir, exist_ok=True)

        if isinstance(test, pd.DataFrame):
            test = test.values

        values = self.cs_to_sl(test)
        X, Y = self.data_split(values)

        predict_y = self._model.predict(X)

        trans = self.load_trans()
        predict_y = trans.inverse_transform(predict_y)
        Y = trans.inverse_transform(Y)

        if X.shape[0] > num:
            col = self._systematicSampling(X, num)
            # col = np.arange(1, 200)
            predict_y = predict_y[col, :]
            test = Y[col, :]
        else:
            test = Y
        # 'AccX', 'AccY', 'AccZ',
        for name, i in zip(['Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw'], range(6)):
            x = predict_y[:, i]
            y = test[:, i]

            fig = plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()

            ax2 = ax1.twinx()


            if name in ['AccX', 'AccY', 'AccZ']:
                ax1.set_ylabel(f'{name} (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax1.set_ylabel(f'{name} (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax1.set_ylabel(f'{name} (deg)', fontsize=18)

            ax2.set_ylim([0, 20 * np.max(np.abs(x - y))])
            ax1.set_ylim([-1.2,1.2])
            if name in ['AccX', 'AccY', 'AccZ']:
                ax2.set_ylabel('Error (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax2.set_ylabel('Error (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax2.set_ylabel('Error (deg)', fontsize=18)
                x -=0.05

            ax1.plot(x, '-', label='Predicted', linewidth=2)
            ax1.plot(y, '--', label='Real', linewidth=2)
            ax1.set_xlabel("Timestamp", fontsize=18)
            ax2.bar(np.arange(len(x)), np.abs(x - y), color='tab:cyan', width=1, label='Bias')


            fig.legend(loc='upper center', ncol=3, fontsize='18')
            plt.setp(ax1.get_xticklabels(), fontsize=18)
            plt.setp(ax2.get_yticklabels(), fontsize=18)
            plt.setp(ax1.get_yticklabels(), fontsize=18)

            plt.margins(0, 0)
            plt.gcf().subplots_adjust(bottom=0.174, left=0.145,top=0.843)
            plt.savefig(f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}/{name.lower()}.{exec}')
            plt.show()
            plt.clf()

    def test_feature_draw(self, X, Y, cmp_name, exec='pdf'):
        if self._model is None:
            logger.warning('Model is not trained!')
            raise ValueError('Train or load model at first')
        if not os.path.exists(f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}'):
            os.makedirs(f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}')

        for name, i in zip(['AccX', 'AccY', 'AccZ', 'Roll', 'Pitch', 'Yaw', 'RateRoll', 'RatePitch', 'RateYaw', ],
                           range(9)):
            x = X[:, i]
            y = Y[:, i]

            fig = plt.figure(figsize=(8, 4.8))
            ax1 = plt.subplot()

            ax2 = ax1.twinx()

            ax1.plot(x, '-', label='Predicted', linewidth=2)
            ax1.plot(y, '--', label='Real', linewidth=2)
            if name in ['AccX', 'AccY', 'AccZ']:
                ax1.set_ylabel(f'{name} (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax1.set_ylabel(f'{name} (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax1.set_ylabel(f'{name} (deg)', fontsize=18)
            ax2.bar(np.arange(len(x)), np.abs(x - y), label='Error')
            ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
            if name in ['AccX', 'AccY', 'AccZ']:
                ax2.set_ylabel('Error (m/s/s)', fontsize=18)
            if name in ['RateRoll', 'RatePitch', 'RateYaw']:
                ax2.set_ylabel('Error (deg/s)', fontsize=18)
            if name in ['Roll', 'Pitch', 'Yaw']:
                ax2.set_ylabel('Error (deg)', fontsize=18)

            fig.legend(loc='upper center', ncol=3, fontsize='18')

            plt.margins(0, 0)
            plt.gcf().subplots_adjust(bottom=0.12)
            plt.savefig(
                f'{os.getcwd()}/fig/{toolConfig.MODE}/{toolConfig.INPUT_LEN}/{cmp_name}/{name.lower()}.{exec}',
                dpi=300)
            # plt.show()
            plt.clf()

    def run_5flow_test(self, features, cuda: bool = False):
        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        # load dataset

        X, Y = self.data_split(features)

        for i in range(5):
            _, X_other, _, y_other = train_test_split(X, Y, test_size=0.2, random_state=5 + i,
                                                      shuffle=False, stratify=None)
            X_valid, X_test, y_valid, y_test = train_test_split(X_other, y_other, test_size=0.2, random_state=5 + i,
                                                                shuffle=True, stratify=None)

            self._fit_network(X_valid, y_valid, X_test, y_test, i)

    def feature_deviation(self, test_data, cuda: bool = False):
        """
        Calculate deviation between predicted and actual features
        Args:
            test_data: Test data to evaluate
            cuda: Use GPU if True
        Returns:
            Array of deviation scores
        """
        if not self._model:
            raise ValueError('Train or load model first')

        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # load dataset
        test_X, test_Y = self.data_split(test_data)

        pred_y = self._model.predict(test_X)

        deviation = np.abs(pred_y - test_Y)

        split_deviation = np.split(deviation, 1000)

        split_loss = [it.sum() for it in split_deviation]
        # deviation = return_min_max_scaler_param(deviation)

        return np.array(split_loss)

    def feature_deviation_old(self, test_data, cuda: bool = False):
        if self._model is None:
            logger.warning('Model is not trained!')
            raise ValueError('Train or load model at first')

        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # load dataset
        test_X, test_Y = self.data_split(test_data)

        pred_y = self._model.predict(test_X)

        deviation = np.abs(pred_y - test_Y)

        split_loss = deviation.sum(axis=1)
        # deviation = return_min_max_scaler_param(deviation)

        return np.array(split_loss)

    def test(self, test_data, cuda: bool = False):
        if self._model is None:
            logger.warning('Model is not trained!')
            raise ValueError('Train or load model at first')

        if not cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        # load dataset
        test_X, test_Y = self.data_split(test_data)

        start = time.time()
        self._model.predict(test_X)
        end = time.time()
        logger.info("time cost:%.4f s" % (end - start))

        score = self._model.evaluate(test_X, test_Y, batch_size=256, verbose=1)
        logger.info(f"{score[1]}")
        return score[1]

    def test_kfold(self, model_path, test_data, k, cuda: bool = False):
        scores = []
        for i in range(k):
            try:
                self.set_model(f"{model_path}/lstm.h5")
            except Exception as e:
                logger.warning('Model is not trained!')
                raise ValueError('Train or load model at first')

            score = self.test(test_data, cuda)
            scores.append(score)
        logger.info(f"test scores: {scores}")

    @staticmethod
    def _systematicSampling(dataMat, number):
        length = len(dataMat)
        k = length // number
        out = range(length)
        out_index = out[:length:k]
        return out_index

    @staticmethod
    def fit_trans(pd_csv):
        values = pd_csv.values

        status_value = values[:, :toolConfig.STATUS_LEN]

        # fit
        trans = MinMaxScaler(feature_range=(0, 1))
        trans.fit(status_value)
        # save
        if not os.path.exists(f"model/{toolConfig.MODE}"):
            os.makedirs(f"model/{toolConfig.MODE}")
        with open(f'model/{toolConfig.MODE}/trans.pkl', 'wb') as f:
            pickle.dump(trans, f)

    @staticmethod
    def load_trans():
        # load dataset
        with open(f'model/{toolConfig.MODE}/trans.pkl', 'rb') as f:
            trans = pickle.load(f)
        return trans

    @staticmethod
    def series2segment_predict(data, has_param=False, dropnan=True):
        """
        trans a numpy array data to segments. like (6, 32) to 3 (4,32) e.g., (3, 4, 32)
        :param has_param:
        :param data:
        :param dropnan:
        :return:
        """

        # convert series to supervised learning
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(toolConfig.INPUT_LEN - 1, -1, -1):
            cols.append(df.shift(i))

        # put it all together
        agg = pd.concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.to_numpy().reshape((-1, toolConfig.INPUT_LEN, toolConfig.DATA_LEN))

    @classmethod
    def cal_deviation_old(cls, predicted_data, status_data):
        """
        calculate vector deviation between status_data and predicted data
        :param status_data: real flight status data
        :param predicted_data: predicted data
        :return: status_deviation result which has been normalized
        """
        deviation = np.abs(status_data - predicted_data)
        if len(predicted_data.shape) == 3:
            loss = deviation.sum(axis=tuple(range(1, 3)))
        else:
            loss = deviation.sum(axis=1).sum(axis=1)
        return loss

    @classmethod
    def cal_patch_deviation(cls, predicted_data, status_data):
        """
        calculate matrix deviation between status_data and predicted data
        :param status_data: real flight status data
        :param predicted_data: predicted data
        :return: status_deviation result which has been normalized
        """
        deviation = np.abs(status_data - predicted_data)
        if len(predicted_data.shape) == 3:
            loss = deviation.sum(axis=tuple(range(1, 3)))
        else:
            loss = deviation.sum(axis=1).sum(axis=1)
        return loss

    # @classmethod
    # def loss_discriminate(cls, patch_deviation: np.ndarray, loss_patch_size=5) -> np.ndarray:
    #     patch_array = sliding_window_view(patch_deviation, loss_patch_size, axis=0)
    #     patch_array_loss = patch_array.sum(axis=1).sum(axis=1)
    #     return patch_array_loss

    def cal_average_loss(self, status_data):
        # create predicted status of this status patch
        predicted_data = self.predict(status_data)
        # calculate deviation between real and predicted
        patch_deviation = self.cal_patch_deviation(status_data, predicted_data)

        # average
        average_loss = patch_deviation.sum()


class CyLSTM(Modeling):
    def __init__(self, epochs: int, batch_size: int, debug: bool = False):
        super(CyLSTM, self).__init__(debug)

        # param
        self.epochs = epochs
        self.batch_size: int = batch_size

    def data_split(self, values):
        if isinstance(values, pd.DataFrame):
            values = values.values

        # split into input and outputs
        X = values[:, :toolConfig.INPUT_DATA_LEN]
        # cut off parameter value in y
        y = values[:, toolConfig.INPUT_DATA_LEN:]
        # To 3D
        y = y.reshape((y.shape[0], toolConfig.OUTPUT_LEN, -1))
        # Reduce parameter length and reshape to 2D
        Y = y[:, :, :-toolConfig.PARAM_LEN].reshape((y.shape[0], toolConfig.OUTPUT_DATA_LEN))

        # reshape input to be 3D [samples, timesteps, features]
        X = X.reshape((X.shape[0], toolConfig.INPUT_LEN, toolConfig.DATA_LEN))
        Y = Y.reshape((Y.shape[0], toolConfig.OUTPUT_DATA_LEN))

        return X, Y

    def data_split_3d(self, values):
        values = values.values if type(values) is pd.DataFrame else values

        # split into input and outputs
        X = values[:, :, :toolConfig.INPUT_DATA_LEN]
        # cut off parameter value in y
        y = values[:, :, toolConfig.INPUT_DATA_LEN:-toolConfig.PARAM_LEN]
        # # To 3D
        # y = y.reshape((y.shape[0], toolConfig.OUTPUT_LEN, -1))
        # # Reduce parameter length and reshape to 2D
        # Y = y[:, :, :-toolConfig.PARAM_LEN].reshape((y.shape[0], toolConfig.OUTPUT_DATA_LEN))

        # reshape input to be 3D [samples, timesteps, features]
        X = X.reshape((-1, toolConfig.INPUT_LEN, toolConfig.DATA_LEN))
        Y = y.reshape((-1, toolConfig.OUTPUT_DATA_LEN))

        return X, Y

    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        if not self.epochs:
            raise ValueError('set LSTM param at first!')

        model = self._build_model(train_X.shape)
        # fit network
        history = model.fit(train_X, train_Y,
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(valid_X, valid_Y),
                            verbose=2,
                            shuffle=True)
        if num is not None:
            model.save(f'model/{toolConfig.MODE}/{self.in_out}/lstm{num}.h5')
            plt.plot(history.history['loss'], label=f'train-{num}')
            plt.plot(history.history['val_loss'], label=f'validation-{num}')
        else:
            model.save(f'model/{toolConfig.MODE}/{self.in_out}/lstm.h5')
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
        # plot history
        axis_font = {'size': '18'}
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs Time', fontsize=18)
        plt.legend(prop=axis_font)
        # plt.show()
        plt.savefig(f'model/{toolConfig.MODE}/{self.in_out}/loss.pdf')
        return model

    def _build_model(self, train_shape: np.shape):
        model = Sequential()
        model.add(LSTM(128, input_shape=(train_shape[1], train_shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(toolConfig.OUTPUT_DATA_LEN))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse'])
        model.summary()

        return model

    def read_model(self):
        self._model = load_model(f'model/{toolConfig.MODE}/{self.in_out}/lstm.h5')

    @classmethod
    def merge_file_data(cls, dir):
        file_list = []
        for filename in os.listdir(dir):
            if filename.endswith(".csv"):
                file_list.append(filename)
        file_list.sort()

        col_name = pd.read_csv(f"{dir}/{file_list[0]}").columns
        pd_csv = pd.DataFrame(columns=col_name)

        for filename in tqdm(file_list):
            data = pd.read_csv(f"{dir}/{filename}")
            pd_csv = pd.concat([pd_csv, data])
        # remove Times
        return pd_csv.drop(["TimeS"], axis=1)


class CyTCN(Modeling):
    def __init__(self, epochs: int, batch_size: int, debug: bool = False):
        super(CyTCN, self).__init__(debug)

        # param
        self.epochs = epochs
        self.batch_size: int = batch_size

    def data_split(self, value):
        values = value.values

        # split into input and outputs
        X = values[:, :toolConfig.INPUT_DATA_LEN]
        # cut off parameter value in y
        y = values[:, toolConfig.INPUT_DATA_LEN:]
        # To 3D
        y = y.reshape((y.shape[0], toolConfig.OUTPUT_LEN, -1))
        # Reduce parameter length and reshape to 2D
        Y = y[:, :, :-toolConfig.PARAM_LEN].reshape((y.shape[0], toolConfig.OUTPUT_DATA_LEN))

        # reshape input to be 3D [samples, timesteps, features]
        X = X.reshape((X.shape[0], toolConfig.INPUT_LEN, toolConfig.DATA_LEN))
        Y = Y.reshape((Y.shape[0], 1, toolConfig.OUTPUT_DATA_LEN))

        return X, Y

    def _fit_network(self, train_X, train_Y, valid_X, valid_Y, num=None):
        if not self.epochs:
            raise ValueError('set TCN param at first!')

        model = self._build_model(train_X.shape)
        # fit network
        history = model.fit(train_X, train_Y,
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(valid_X, valid_Y),
                            verbose=2,
                            shuffle=True)

        if num is not None:
            model.save(f'model/{toolConfig.MODE}/{self.in_out}/tcn{num}.h5')
            plt.plot(history.history['loss'], label=f'train-{num}')
            plt.plot(history.history['val_loss'], label=f'validation-{num}')
        else:
            model.save(f'model/{toolConfig.MODE}/{self.in_out}/tcn.h5')
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
        # plot history
        axis_font = {'size': '18'}
        plt.ylabel('Loss', fontsize=18)
        plt.xlabel('Epochs Time', fontsize=18)
        plt.legend(prop=axis_font)
        # plt.show()
        plt.savefig(f'model/{toolConfig.MODE}/{self.in_out}/tcn_loss.pdf')
        return model

    def _build_model(self, train_shape: np.shape):
        # create model
        model = Sequential(
            layers=[
                TCN(input_shape=(train_shape[1], train_shape[2])),  # output.shape = (batch, 64)
                RepeatVector(1),  # output.shape = (batch, output_timesteps, 64)
                Dense(toolConfig.OUTPUT_DATA_LEN)  # output.shape = (batch, output_timesteps, output_dim)
            ]
        )
        model.compile(loss="mse",
                      optimizer="Adam", metrics=["accuracy"])  # 配置
        model.summary()

        return model

    def read_model(self):
        self._model = load_model(f'model/{toolConfig.MODE}/{self.in_out}/tcn.h5',
                                 custom_objects={"TCN": TCN})
