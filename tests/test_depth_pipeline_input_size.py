import importlib.util
from pathlib import Path
from types import SimpleNamespace


BACKEND_PATH = Path(__file__).resolve().parents[1] / "depth_module" / "depth" / "backends" / "depth_anything_v2_small.py"
SPEC = importlib.util.spec_from_file_location("depth_anything_v2_small_test", BACKEND_PATH)
assert SPEC is not None and SPEC.loader is not None
BACKEND = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND)
DepthAnythingV2Small = BACKEND.DepthAnythingV2Small


def test_pipeline_processor_uses_configured_input_size(monkeypatch):
    image_processor = SimpleNamespace(size={"height": 518, "width": 518})
    pipeline = lambda **_kwargs: SimpleNamespace(image_processor=image_processor)
    monkeypatch.setitem(__import__("sys").modules, "transformers", SimpleNamespace(pipeline=pipeline))

    backend = DepthAnythingV2Small(device="cpu", input_size=252)
    backend._load()

    assert image_processor.size == {"height": 252, "width": 252}
