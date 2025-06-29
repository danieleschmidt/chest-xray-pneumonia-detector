import importlib
import sys
import types

import pytest


def load_train_engine(monkeypatch):
    callbacks_stub = types.SimpleNamespace(
        ModelCheckpoint=lambda *a, **k: None,
        EarlyStopping=lambda *a, **k: None,
        ReduceLROnPlateau=lambda *a, **k: None,
    )
    keras_stub = types.SimpleNamespace(
        callbacks=callbacks_stub,
        optimizers=types.SimpleNamespace(Adam=lambda *a, **k: None),
        utils=types.SimpleNamespace(
            image_dataset_from_directory=lambda *a, **k: None,
            to_categorical=lambda *a, **k: None,
        ),
    )
    tf_stub = types.SimpleNamespace(keras=keras_stub, random=types.SimpleNamespace(set_seed=lambda *a, **k: None))
    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.callbacks", keras_stub.callbacks)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", keras_stub.optimizers)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.utils", keras_stub.utils)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.preprocessing", types.ModuleType("preproc"))
    img_module = types.ModuleType("image")
    img_module.ImageDataGenerator = object
    monkeypatch.setitem(sys.modules, "tensorflow.keras.preprocessing.image", img_module)

    metrics_stub = types.ModuleType("metrics")
    metrics_stub.precision_score = lambda *a, **k: 0
    metrics_stub.recall_score = lambda *a, **k: 0
    metrics_stub.f1_score = lambda *a, **k: 0
    metrics_stub.roc_auc_score = lambda *a, **k: 0
    metrics_stub.confusion_matrix = lambda *a, **k: [[0]]
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_stub)

    cw_stub = types.SimpleNamespace(compute_class_weight=lambda *a, **k: [1, 1])
    utils_stub = types.ModuleType("utils")
    utils_stub.class_weight = cw_stub
    monkeypatch.setitem(sys.modules, "sklearn.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "sklearn.utils.class_weight", cw_stub)

    monkeypatch.setitem(sys.modules, "PIL", types.ModuleType("PIL"))
    monkeypatch.setitem(sys.modules, "PIL.Image", types.ModuleType("Image"))

    np_stub = types.SimpleNamespace(unique=lambda x: sorted(set(x)))
    monkeypatch.setitem(sys.modules, "numpy", np_stub)

    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", types.ModuleType("pyplot"))
    monkeypatch.setitem(sys.modules, "seaborn", types.ModuleType("seaborn"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))

    dl_stub = types.ModuleType("dl")
    dl_stub.create_data_generators = lambda **k: ("train", "val")
    monkeypatch.setitem(sys.modules, "src.data_loader", dl_stub)

    mb_stub = types.ModuleType("mb")
    mb_stub.create_simple_cnn = lambda **k: "simple"
    mb_stub.create_transfer_learning_model = lambda **k: "transfer"
    mb_stub.create_cnn_with_attention = lambda **k: "attention"
    monkeypatch.setitem(sys.modules, "src.model_builder", mb_stub)

    if "src.train_engine" in sys.modules:
        del sys.modules["src.train_engine"]
    return importlib.import_module("src.train_engine")


def test_create_model_variants(monkeypatch):
    te = load_train_engine(monkeypatch)
    args = te.TrainingArgs(use_attention_model=True)
    assert te._create_model(args, (64, 64, 3)) == "attention"

    args = te.TrainingArgs(use_transfer_learning=True, use_attention_model=False)
    assert te._create_model(args, (64, 64, 3)) == "transfer"

    args = te.TrainingArgs(use_transfer_learning=False)
    assert te._create_model(args, (64, 64, 3)) == "simple"


def test_compute_class_weights_manual(monkeypatch):
    te = load_train_engine(monkeypatch)
    dummy_gen = types.SimpleNamespace(classes=[0, 1])
    args = te.TrainingArgs(num_classes=2, class_weights=[0.3, 0.7])
    weights = te._compute_class_weights(dummy_gen, args)
    assert weights == {0: 0.3, 1: 0.7}


def test_compute_class_weights_auto(monkeypatch):
    te = load_train_engine(monkeypatch)
    dummy_gen = types.SimpleNamespace(classes=[0, 0, 1, 1])
    args = te.TrainingArgs(num_classes=2)
    weights = te._compute_class_weights(dummy_gen, args)
    assert weights == {0: 1, 1: 1}


def test_compute_class_weights_invalid(monkeypatch):
    te = load_train_engine(monkeypatch)
    dummy_gen = types.SimpleNamespace(classes=[0])
    args = te.TrainingArgs(num_classes=2, class_weights=[1.0])
    with pytest.raises(ValueError):
        te._compute_class_weights(dummy_gen, args)

