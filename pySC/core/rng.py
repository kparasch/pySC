from pydantic import BaseModel, field_serializer, model_validator, PrivateAttr
from typing import Union, Optional
from numpy.random import default_rng
import numpy as np

class RNG(BaseModel):
    seed: int
    rng_state: Optional[dict[str, Union[str, int, dict[str, int]]]] = None
    default_truncation: Optional[float] = None
    _rng = PrivateAttr(default=None)

    @field_serializer('rng_state')
    def update_rng_state(self, rng_state: Optional[dict[str, Union[str, int, dict[str, int]]]]):
        return self._rng.bit_generator.state.copy()

    @model_validator(mode="after")
    def initialize_rng(self):
        self._rng = default_rng(seed=self.seed)
        if self.rng_state is None:
            self.rng_state = self._rng.bit_generator.state.copy()
        else:
            self._rng.bit_generator.state = self.rng_state
        return self

    def normal_trunc(self, loc: float = 0, scale: float = 1,
                     sigma_truncate: Optional[float] = None,
                     size: Optional[int]=None) -> Union[float, np.ndarray]:
        if sigma_truncate is None:
            sigma_truncate = self.default_truncation

        ## if still None then don't truncate
        if sigma_truncate is None:
            return self._rng.normal(loc, scale, size=size)
        else:
            if size is not None:
                ## TODO: optimize this ?
                return np.array([self.normal_trunc(loc, scale, sigma_truncate) for _ in range(size)])
            ret = self._rng.normal()
            while abs(ret) > sigma_truncate:
                ret = self._rng.normal()
            return loc + ret * scale

    def normal(self, loc: float = 0, scale: float = 1, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return self._rng.normal(loc=loc, scale=scale, size=size)

    def uniform(self, low: float = 0, high: float = 1, size: Optional[int] = None) -> Union[float, np.ndarray]:
        return low + self._rng.random(size=size) * (high - low)

    def randomize_rng(self) -> None:
        self._rng = default_rng()
