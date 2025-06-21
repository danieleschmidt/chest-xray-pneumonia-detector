import pytest

# Skip tests if TensorFlow is unavailable
tf = pytest.importorskip("tensorflow")

from src.model_builder import create_simple_cnn, create_cnn_with_attention


def test_simple_cnn_output_shape():
    model = create_simple_cnn((32, 32, 3))
    assert model.output_shape == (None, 1)


def test_attention_cnn_output_shape():
    model = create_cnn_with_attention((32, 32, 3))
    assert model.output_shape == (None, 1)
