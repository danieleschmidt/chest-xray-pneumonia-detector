import importlib
import sys
import types


def load_data_split(monkeypatch):
    def tts(data, test_size=None, random_state=None):
        return data[:-1], data[-1:]

    sklearn_stub = types.SimpleNamespace(model_selection=types.SimpleNamespace(train_test_split=tts))
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_stub)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", sklearn_stub.model_selection)
    if "src.data_split" in sys.modules:
        del sys.modules["src.data_split"]
    return importlib.import_module("src.data_split")


def test_split_dataset_stub(monkeypatch, tmp_path):
    ds = load_data_split(monkeypatch)
    input_dir = tmp_path / "input"
    for cls in ["a", "b"]:
        cls_dir = input_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            (cls_dir / f"img{i}.jpg").write_text("x")

    output_dir = tmp_path / "output"
    ds.split_dataset(str(input_dir), str(output_dir), val_frac=0.25, test_frac=0.25)

    for split in ["train", "val", "test"]:
        for cls in ["a", "b"]:
            assert (output_dir / split / cls).exists()

    assert len(list((output_dir / "train" / "a").glob("*.jpg"))) == 1
    assert len(list((output_dir / "val" / "a").glob("*.jpg"))) == 1
    assert len(list((output_dir / "test" / "a").glob("*.jpg"))) == 1
