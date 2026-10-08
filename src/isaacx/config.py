"""Typed loader for ``configs/*.yaml``.

The YAML is the contract between the sim, detection, tracking and inference stages. Loading it
through these models catches typos and out-of-range values at startup instead of mid-run:
unknown keys are rejected, thresholds must be probabilities, and the model input size must be a
multiple of YOLO's 32-pixel stride.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, field_validator

Vec3 = tuple[float, float, float]


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SimulationConfig(Section):
    headless: bool
    physics_dt: PositiveFloat
    rendering_dt: PositiveFloat
    usd_scene: str


class DetectionConfig(Section):
    model: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    nms_threshold: float = Field(ge=0.0, le=1.0)
    target_classes: list[str] = Field(min_length=1)


class TrackingConfig(Section):
    algorithm: str
    max_age: PositiveInt
    min_hits: PositiveInt
    iou_threshold: float = Field(ge=0.0, le=1.0)


class RobotConfig(Section):
    type: str
    controller: str
    end_effector: str


class CameraConfig(Section):
    width: PositiveInt
    height: PositiveInt
    fov: float = Field(gt=0.0, lt=180.0)
    position: Vec3
    target: Vec3


class InferenceConfig(Section):
    onnx_path: str
    imgsz: PositiveInt
    rust_binary: str

    @field_validator("imgsz")
    @classmethod
    def check_stride(cls, value: int) -> int:
        if value % 32:
            raise ValueError(f"imgsz must be a multiple of 32 (YOLO stride), got {value}")
        return value


class Config(Section):
    simulation: SimulationConfig
    detection: DetectionConfig
    tracking: TrackingConfig
    robot: RobotConfig
    camera: CameraConfig
    inference: InferenceConfig


def load_config(path: str | Path) -> Config:
    """Read and validate a config YAML file.

    Raises ``FileNotFoundError`` if the file is missing and ``pydantic.ValidationError`` if it
    does not match the schema.
    """
    with Path(path).open(encoding="utf-8") as fh:
        return Config.model_validate(yaml.safe_load(fh))
