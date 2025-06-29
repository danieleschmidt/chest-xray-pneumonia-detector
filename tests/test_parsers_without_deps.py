import importlib
import sys
import types


def test_train_engine_parse_args(monkeypatch):
    callbacks_stub = types.SimpleNamespace(
        ModelCheckpoint=lambda *a, **k: None,
        EarlyStopping=lambda *a, **k: None,
        ReduceLROnPlateau=lambda *a, **k: None,
    )
    tf_stub = types.SimpleNamespace(
        keras=types.SimpleNamespace(
            callbacks=callbacks_stub,
            optimizers=types.SimpleNamespace(Adam=lambda *a, **k: None),
            utils=types.SimpleNamespace(
                image_dataset_from_directory=lambda *a, **k: None
            ),
        )
    )
    # Provide minimal modules required at import time
    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", tf_stub.keras)
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras.callbacks", tf_stub.keras.callbacks
    )
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras.optimizers", tf_stub.keras.optimizers
    )
    monkeypatch.setitem(sys.modules, "tensorflow.keras.utils", tf_stub.keras.utils)
    metrics_stub = types.ModuleType("metrics")
    metrics_stub.precision_score = lambda *a, **k: 0
    metrics_stub.recall_score = lambda *a, **k: 0
    metrics_stub.f1_score = lambda *a, **k: 0
    metrics_stub.roc_auc_score = lambda *a, **k: 0
    metrics_stub.confusion_matrix = lambda *a, **k: [[0]]
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_stub)
    utils_stub = types.ModuleType("utils")
    class_weight_mod = types.SimpleNamespace(
        compute_class_weight=lambda *a, **k: [1, 1]
    )
    utils_stub.class_weight = class_weight_mod
    monkeypatch.setitem(sys.modules, "sklearn.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "sklearn.utils.class_weight", class_weight_mod)
    monkeypatch.setitem(sys.modules, "PIL", types.ModuleType("PIL"))
    monkeypatch.setitem(sys.modules, "PIL.Image", types.ModuleType("Image"))
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", types.ModuleType("pyplot"))
    monkeypatch.setitem(sys.modules, "seaborn", types.ModuleType("seaborn"))
    monkeypatch.setitem(sys.modules, "pandas", types.ModuleType("pandas"))
    tf_stub.keras.utils.to_categorical = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    monkeypatch.setitem(
        sys.modules,
        "tensorflow.keras.preprocessing",
        types.ModuleType("preproc"),
    )
    img_module = types.ModuleType("image")
    img_module.ImageDataGenerator = object
    monkeypatch.setitem(
        sys.modules,
        "tensorflow.keras.preprocessing.image",
        img_module,
    )
    dl_stub_local = types.ModuleType("d")
    dl_stub_local.create_data_generators = lambda *a, **k: (None, None)
    monkeypatch.setitem(sys.modules, "src.data_loader", dl_stub_local)
    mb_stub_local = types.ModuleType("m")
    mb_stub_local.create_simple_cnn = lambda *a, **k: None
    mb_stub_local.create_transfer_learning_model = lambda *a, **k: None
    mb_stub_local.create_cnn_with_attention = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "src.model_builder", mb_stub_local)

    te = importlib.import_module("src.train_engine")
    args = te._parse_args(["--epochs", "3"])
    assert isinstance(args, te.TrainingArgs)
    assert args.epochs == 3


def test_pipeline_parse_args(monkeypatch):
    tf_stub = types.ModuleType("tensorflow")
    monkeypatch.setitem(sys.modules, "tensorflow", tf_stub)
    dl_stub = types.ModuleType("data_loader")
    dl_stub.create_data_generators = lambda *a, **k: (None, None)
    monkeypatch.setitem(sys.modules, "data_loader", dl_stub)
    mb_stub = types.ModuleType("model_builder")
    mb_stub.create_simple_cnn = lambda *a, **k: None
    mb_stub.create_transfer_learning_model = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "model_builder", mb_stub)
    from src.chest_xray_pneumonia_detector import pipeline

    cfg = pipeline._parse_args(
        [
            "--train_dir",
            "x",
            "--val_dir",
            "y",
            "--epochs",
            "2",
        ]
    )
    assert cfg.epochs == 2
    assert cfg.train_dir == "x"
    assert cfg.val_dir == "y"
