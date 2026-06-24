"""Tests for the PyTorch surrogate model.

Gated on torch being importable. In a minimal environment the module is
skipped, so the suite stays green.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

import icsearcher.config as cfg  # noqa: E402
from icsearcher.model import CyLSTM, CyTCN, _LSTMNet, make_predictor  # noqa: E402


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


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_make_predictor_selects_by_model_type(monkeypatch, mode_name):
    """make_predictor returns CyLSTM/CyTCN per the config's MODEL_TYPE."""
    monkeypatch.delenv("ICSEARCHER_MODEL_TYPE", raising=False)
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)

    fresh.__dict__["MODEL_TYPE"] = "lstm"
    assert isinstance(make_predictor(2, 4), CyLSTM)

    fresh.__dict__["MODEL_TYPE"] = "tcn"
    assert isinstance(make_predictor(2, 4), CyTCN)


def test_make_predictor_rejects_unknown_type(monkeypatch):
    """An unsupported MODEL_TYPE raises a clear ValueError."""
    fresh = cfg.ToolConfig(mode="Ardupilot")
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    fresh.__dict__["MODEL_TYPE"] = "gru"
    with pytest.raises(ValueError, match="Unknown MODEL_TYPE"):
        make_predictor(2, 4)


@pytest.mark.parametrize("mode_name", ["Ardupilot", "PX4"])
def test_cytcn_predict_after_train(monkeypatch, mode_name):
    """A trained TCN predicts finite, correctly-shaped output.

    Guards against the Y-shape bug that used to break TCN training: the
    network's forward() returns 2-D (batch, OUTPUT_DATA_LEN), so data_split
    must produce a 2-D Y too or MSELoss broadcasts to (batch, batch, O).
    """
    fresh = cfg.ToolConfig(mode=mode_name)
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    rng = np.random.default_rng(0)
    n = 64
    width = fresh.INPUT_DATA_LEN + fresh.OUTPUT_DATA_LEN + fresh.PARAM_LEN
    import pandas as pd
    df = pd.DataFrame(rng.uniform(-1, 1, size=(n, width)))

    tcn = CyTCN(epochs=2, batch_size=16)
    tcn._device = torch.device("cpu")
    # Y must be 2-D so MSELoss doesn't broadcast; verify the shape explicitly.
    X, Y = tcn.data_split(df)
    assert Y.shape == (n, fresh.OUTPUT_DATA_LEN)
    tcn.train(df, cuda=False)
    assert tcn._model is not None

    pred = tcn.predict_feature(X)
    assert pred.shape == (n, fresh.OUTPUT_DATA_LEN)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("cls", [CyLSTM, CyTCN])
def test_data_split_3d_inherited_by_both_surrogates(monkeypatch, cls):
    """Both surrogate subclasses inherit data_split_3d from the base class.

    Fuzzing (search/problem.py) calls predictor.data_split_3d(...), so it must
    exist on whatever surrogate MODEL_TYPE selects.
    """
    fresh = cfg.ToolConfig(mode="Ardupilot")
    monkeypatch.setattr(cfg, "toolConfig", fresh)
    rng = np.random.default_rng(1)
    n_params, n_windows = 5, 7
    width = fresh.INPUT_DATA_LEN + fresh.OUTPUT_DATA_LEN + fresh.PARAM_LEN
    values = rng.uniform(-1, 1, size=(n_params, n_windows, width))
    X, Y = cls(1, 1).data_split_3d(values)
    assert X.shape == (n_params * n_windows, fresh.INPUT_LEN, fresh.DATA_LEN)
    assert Y.shape == (n_params * n_windows, fresh.OUTPUT_DATA_LEN)
