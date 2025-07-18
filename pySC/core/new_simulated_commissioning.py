from pydantic import BaseModel, model_validator, Field
from typing import Optional

from .lattice import ATLattice, XSuiteLattice
from .magnetsettings import MagnetSettings
from .supports import SupportSystem
from .bpm_system import BPMSystem
from .magnet import MAGNET_NAME_TYPE
from .rfsettings import RFSettings
from ..tuning.tuning_core import Tuning
from .injection import InjectionSettings
from .rng import RNG
from ..control_system.server import start_server as _start_server

class SimulatedCommissioning(BaseModel, extra="forbid"):
    lattice: ATLattice | XSuiteLattice
    magnet_settings: MagnetSettings = MagnetSettings()
    design_magnet_settings: MagnetSettings = MagnetSettings()
    support_system: SupportSystem = SupportSystem()
    bpm_system: BPMSystem = BPMSystem()
    rf_settings: RFSettings = RFSettings()
    injection: InjectionSettings = InjectionSettings()
    tuning: Tuning = Tuning()

    configuration: dict = {}
    magnet_arrays: dict[str, list[MAGNET_NAME_TYPE]] = {}
    control_arrays: dict[str, list[str]] = {}
    seed: int = Field(default=1, frozen=True)
    rng: Optional[RNG] = None

    @model_validator(mode="after")
    def initialize(self):
        self.propagate_parents()
        self.rng = RNG(seed=self.seed)
        return self

    @classmethod
    def from_json(cls, json_str: str) -> "SimulatedCommissioning":
        """
        Load the SimulatedCommissioning instance from a JSON string.
        """
        return cls.model_validate_json(json_str)

    def propagate_parents(self) -> None:
        self.magnet_settings._parent = self
        self.design_magnet_settings._parent = self
        self.support_system._parent = self
        self.bpm_system._parent = self
        self.rf_settings._parent = self
        self.injection._parent = self
        self.tuning._parent = self
        return

    def start_server(self, port : int = 13131, timeout: int = 1) -> None:
        _start_server(self, port=port, refresh_rate=timeout)
