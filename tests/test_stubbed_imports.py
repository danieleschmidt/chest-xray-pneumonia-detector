import importlib
import sys
import types


def stub_tf(monkeypatch):
    class DummyModel:
        trainable = False

        def __call__(self, *args, **kwargs):
            return None

        def compile(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            pass

        def summary(self):
            pass

        def load_weights(self, *args, **kwargs):
            pass

    model_obj = DummyModel()

    class DummyKerasModel:
        def __init__(self, *args, **kwargs):
            pass

        def compile(self, *args, **kwargs):
            pass

    keras = types.SimpleNamespace(
        models=types.SimpleNamespace(
            Sequential=lambda *a, **k: model_obj, Model=DummyKerasModel
        ),
        layers=types.SimpleNamespace(
            Conv2D=lambda *a, **k: (lambda x: x),
            MaxPooling2D=lambda *a, **k: (lambda x: x),
            Flatten=lambda *a, **k: (lambda x: x),
            Dense=lambda *a, **k: (lambda x: x),
            Dropout=lambda *a, **k: (lambda x: x),
            GlobalAveragePooling2D=lambda *a, **k: (lambda x: x),
            Reshape=lambda *a, **k: (lambda x: x),
            multiply=lambda *a, **k: (lambda x: x),
        ),
        optimizers=types.SimpleNamespace(Adam=lambda *a, **k: None),
        losses=types.SimpleNamespace(
            BinaryCrossentropy=lambda *a, **k: None,
            CategoricalCrossentropy=lambda *a, **k: None,
        ),
        applications=types.SimpleNamespace(
            MobileNetV2=lambda *a, **k: model_obj,
            VGG16=lambda *a, **k: model_obj,
        ),
        Input=lambda *a, **k: None,
    )
    tf_module = types.SimpleNamespace(keras=keras)
    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", keras.models)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.layers", keras.layers)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.optimizers", keras.optimizers)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.losses", keras.losses)
    monkeypatch.setitem(
        sys.modules, "tensorflow.keras.applications", keras.applications
    )
    return model_obj


def test_model_builder_import(monkeypatch):
    stub_tf(monkeypatch)
    mb = importlib.import_module("src.model_builder")
    model = mb.create_transfer_learning_model((64, 64, 3))
    assert model is not None
