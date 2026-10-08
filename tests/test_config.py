from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from isaacx.config import Config, load_config

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def default_dict() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_default_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG)
    assert config.detection.confidence_threshold == 0.5
    assert config.detection.nms_threshold == 0.45
    assert config.inference.imgsz == 640
    assert config.camera.position == (1.5, 0.0, 1.2)


def test_unknown_key_is_rejected() -> None:
    data = default_dict()
    data["detection"]["confidance_threshold"] = 0.5  # typo
    with pytest.raises(ValidationError, match="confidance_threshold"):
        Config.model_validate(data)


def test_threshold_must_be_a_probability() -> None:
    data = default_dict()
    data["detection"]["confidence_threshold"] = 1.5
    with pytest.raises(ValidationError, match="confidence_threshold"):
        Config.model_validate(data)


def test_imgsz_must_match_yolo_stride() -> None:
    data = default_dict()
    data["inference"]["imgsz"] = 650
    with pytest.raises(ValidationError, match="multiple of 32"):
        Config.model_validate(data)


def test_camera_position_must_be_3d() -> None:
    data = default_dict()
    data["camera"]["position"] = [1.0, 2.0]
    with pytest.raises(ValidationError, match="position"):
        Config.model_validate(data)


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(DEFAULT_CONFIG.with_name("missing.yaml"))
