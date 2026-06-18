"""Node contract tests — IS_CHANGED, IMAGE validation, inference_mode."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_nodes_module():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("wan_animal_nodes", ROOT / "nodes.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ModuleNotFoundError as exc:
        pytest.skip(f"nodes.py dependencies unavailable: {exc}")
    return mod


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed in this interpreter",
)
def test_all_nodes_define_is_changed():
    nodes = _load_nodes_module()
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        assert hasattr(cls, "IS_CHANGED"), f"{name} missing IS_CHANGED"
        out = cls.IS_CHANGED()
        assert isinstance(out, str), f"{name}.IS_CHANGED must return str"
        assert out == out, f"{name}.IS_CHANGED returned NaN"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed in this interpreter",
)
def test_image_nodes_reject_bad_shape():
    import torch

    nodes = _load_nodes_module()
    bad = torch.zeros(3, 64, 64)

    for cls_name in ("AnimalPoseAndDetection", "AnimalPoseDetectionOneToAllAnimation"):
        cls = nodes.NODE_CLASS_MAPPINGS[cls_name]
        extra = {}
        if cls_name == "AnimalPoseDetectionOneToAllAnimation":
            extra = {"align_to": "none", "draw_head": "full"}
        with pytest.raises(ValueError, match="IMAGE"):
            cls().process(
                model={"yolo": None, "vitpose": None, "dataset": "ap10k"},
                images=bad,
                width=832,
                height=480,
                **extra,
            )

    with pytest.raises(ValueError, match="IMAGE"):
        nodes.AnimalPoseAndDetection().process(
            model={"yolo": None, "vitpose": None, "dataset": "ap10k"},
            images=torch.zeros(1, 64, 64, 3),
            width=832,
            height=480,
            retarget_image=torch.zeros(64, 64, 3),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed in this interpreter",
)
def test_process_methods_use_inference_mode():
    nodes = _load_nodes_module()
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        fn = getattr(cls, cls.FUNCTION)
        src = inspect.getsource(fn)
        assert "inference_mode" in src, f"{name}.{cls.FUNCTION} must use torch.inference_mode()"
