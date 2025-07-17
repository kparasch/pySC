from pydantic import BaseModel
from .control import Control

class Tuning(BaseModel, extra="forbid"):
    HCORR: list[Control] = []
    VCORR: list[Control] = []