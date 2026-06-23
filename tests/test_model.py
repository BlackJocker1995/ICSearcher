"""Tests for the PyTorch surrogate model.

Gated on torch being importable. In a minimal environment the module is
skipped, so the suite stays green.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import icsearcher.config as cfg  # noqa: E402
from icsearcher.model import CyLSTM, CyTCN, _LSTMNet  # noqa: E402


def test_lstm_net_forward_shape():
    """The LSTM net maps (batch, INPUT_LEN, DATA_LEN) -> (batch, OUTPUT_DATA_LEN)."""
    fresh = cfg.ToolConfig(mode="Ardupilot")
    net = _LSTMNet(n_features=fresh.DATA_LEN, input_len=fresh.INPUT_LEN,
                   output_len=fresh.OUTPUT_DATA_LEN)
    x = torch.from_numpy(np.random.default_rng(0).uniform(-1, 1,
            size=(8, fresh.INPUT_LEN, fresh.DATA_LEN)).astype("float32"))
    y = net(x)
    assert y.shape == (8, fresh.OUTPUT_DATA_LEN)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_cylstm_predict_after_train(monkeypatch, mode_name):
    """A few training steps then predict_feature yields finite, shaped output."""
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    # Build a tiny dataset matching the supervised layout: INPUT_DATA_LEN +
    # OUTPUT_DATA_LEN + PARAM_LEN columns.
    rng = np.random.default_rng(0)
    n = 64
    width = fresh.INPUT_DATA_LEN + fresh.OUTPUT_DATA_LEN + fresh.PARAM_LEN
    values = rng.uniform(-1, 1, size=(n, width))
    # data_split expects a DataFrame.
    import pandas as pd
    df = pd.DataFrame(values)

    lstm = CyLSTM(epochs=2, batch_size=16)
    # Force CPU for determinism in CI.
    lstm._device = torch.device("cpu")
    lstm.train(df, cuda=False)
    assert lstm._model is not None

    X, _ = lstm.data_split(df)
    pred = lstm.predict_feature(X)
    assert pred.shape[0] == X.shape[0]
    assert pred.shape[1] == fresh.OUTPUT_DATA_LEN
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_cylstm_data_split_3d_shape(monkeypatch, mode_name):
    """data_split_3d produces 3D X and 2D Y of the documented shapes."""
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    rng = np.random.default_rng(1)
    # (n_params, n_windows, DATA_LEN + OUTPUT_DATA_LEN + PARAM_LEN)
    n_params, n_windows = 5, 7
    width = fresh.INPUT_DATA_LEN + fresh.OUTPUT_DATA_LEN + fresh.PARAM_LEN
    values = rng.uniform(-1, 1, size=(n_params, n_windows, width))
    X, Y = CyLSTM(1, 1).data_split_3d(values)
    assert X.shape == (n_params * n_windows, fresh.INPUT_LEN, fresh.DATA_LEN)
    assert Y.shape == (n_params * n_windows, fresh.OUTPUT_DATA_LEN)


def test_cal_patch_deviation_sums_to_scalar_per_candidate():
    """cal_patch_deviation reduces per-candidate deviation to a 1-D array."""
    predicted = np.ones((4, 3, 2))
    actual = np.zeros((4, 3, 2))
    loss = CyLSTM.cal_patch_deviation(predicted, actual)
    assert loss.shape == (4,)
    assert np.allclose(loss, 6.0)  # 3*2 ones per candidate
