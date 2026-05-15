from enum import StrEnum
from pydantic import BeforeValidator, PlainSerializer, BaseModel, PositiveInt
from typing import Annotated, Union, Optional, Literal, Self
import numpy as np
from pathlib import Path

NPARRAY = Annotated[np.ndarray,
                    BeforeValidator(lambda x: np.array(x)),
                    PlainSerializer(lambda x: x.tolist(), return_type=list)
                   ]

class BaseModelWithSave(BaseModel, extra="forbid"):
    def save_as(self, filename: Union[Path, str], indent: Optional[int] = None) -> None:
        if type(filename) is not Path:
            filename = Path(filename)

        data = self.model_dump()
        suffix = filename.suffix
        with open(filename, 'w') as fp:
            if suffix in ['', '.json']:
                import json
                json.dump(data, fp, indent=indent)
            elif suffix == '.yaml':
                import yaml
                yaml.safe_dump(data, fp, indent=indent)
            else:
                raise Exception(f'Unknown file extension: {suffix}.')
        return

class MagnetType(StrEnum):
    norm_dip = "normal_dipole"
    skew_dip = "skew_dipole"
    norm_quad = "normal_quadrupole"
    skew_quad = "skew_quadrupole"
    norm_sext = "normal_sextupole"
    norm_octu = "normal_octupole"
    undefined = "undefined"

    @classmethod
    def from_component_order(cls, component: Literal["A", "B"], order: PositiveInt) -> Self:
        if component == "B":
            if order == 1:
                return MagnetType.norm_dip
            elif order == 2:
                return MagnetType.norm_quad
            elif order == 3:
                return MagnetType.norm_sext
            elif order == 4:
                return MagnetType.norm_octu
        elif component == "A":
            if order == 1:
                return MagnetType.skew_dip
            if order == 2:
                return MagnetType.skew_quad

        return MagnetType.undefined
