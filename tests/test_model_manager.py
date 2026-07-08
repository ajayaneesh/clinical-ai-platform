import pytest

from app.models.inference import DummyInferenceModel
from app.models.manager import ModelManager


def test_get_returns_registered_default():
    model = DummyInferenceModel()
    manager = ModelManager(default_name="default")
    manager.register("default", model)
    assert manager.get() is model
    assert manager.get("default") is model


def test_get_by_explicit_name():
    a, b = DummyInferenceModel(), DummyInferenceModel()
    manager = ModelManager(default_name="a")
    manager.register("a", a)
    manager.register("b", b)  # future: additional models slot in here
    assert manager.get("a") is a
    assert manager.get("b") is b
    assert manager.get() is a  # default


def test_unknown_model_raises():
    manager = ModelManager(default_name="default")
    manager.register("default", DummyInferenceModel())
    with pytest.raises(KeyError):
        manager.get("missing")


def test_names_lists_registered_models():
    manager = ModelManager(default_name="x")
    manager.register("x", DummyInferenceModel())
    manager.register("y", DummyInferenceModel())
    assert manager.names() == ["x", "y"]
    assert manager.default_name == "x"
