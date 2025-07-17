from __future__ import annotations
from pydantic import BaseModel, PrivateAttr
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .magnet import ControlMagnetLink

class LinearConv(BaseModel, extra="forbid"):
    factor: float = 1.0
    offset: float = 0.0

    def transform(self, value: float) -> float:
        return value * self.factor + self.offset

class Control(BaseModel, extra="forbid"):
    name: str
    setpoint: float
    # calibration: LinearConv = LinearConv() # for future use, if needed
    limits: Optional[tuple[float, float]] = None
    _links: Optional[list[ControlMagnetLink]] = PrivateAttr(default=[])

    def check_limits_and_set(self, setpoint: float) -> None:
        """Validate the setpoint against the control's limits."""
        if self.limits is not None:
            lower_limit, upper_limit = self.limits
            if setpoint < lower_limit:
                print(f'WARNING: Setpoint {setpoint} for control "{self.name}" is out of limits ({lower_limit}, {upper_limit})')
                self.setpoint = lower_limit
            elif setpoint > upper_limit:
                print(f'WARNING: Setpoint {setpoint} for control "{self.name}" is out of limits ({lower_limit}, {upper_limit})')
                self.setpoint = upper_limit
            else:
                self.setpoint = setpoint
        else:
            self.setpoint = setpoint
