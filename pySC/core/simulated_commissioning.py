from pydantic import BaseModel, model_validator, Field, PrivateAttr
from typing import Optional
import json
import numpy as np
import logging

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
from .control import KnobData

logger = logging.getLogger(__name__)

class SimulatedCommissioning(BaseModel, extra="forbid"):
    lattice: ATLattice | XSuiteLattice
    magnet_settings: MagnetSettings = MagnetSettings()
    design_magnet_settings: MagnetSettings = MagnetSettings()
    support_system: SupportSystem = SupportSystem()
    bpm_system: BPMSystem = BPMSystem()
    rf_settings: RFSettings = RFSettings()
    design_rf_settings: RFSettings = RFSettings()
    injection: InjectionSettings = InjectionSettings()
    tuning: Tuning = Tuning()

    configuration: dict = {}
    magnet_arrays: dict[str, list[MAGNET_NAME_TYPE]] = {}
    control_arrays: dict[str, list[str]] = {}
    seed: int = Field(default=1, frozen=True)
    rng: Optional[RNG] = None

    _initialized: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def initialize(self):
        if not self._initialized:
            self._initialized = True

            self.propagate_parents()
            if self.rng is None:
                self.rng = RNG(seed=self.seed)
            self.support_system.update_all()
            self.design_magnet_settings.sendall()
            self.magnet_settings.sendall()
            for rf_settings in [self.rf_settings, self.design_rf_settings]:
                for system_name in rf_settings.systems:
                    rf_settings.systems[system_name].trigger_update()
        return self

    @classmethod
    def from_json(cls, json_filename: str, lattice_file: Optional[str] = None) -> "SimulatedCommissioning":
        """
        Load the SimulatedCommissioning instance from a file with a JSON format.
        """
        with open(json_filename, 'r') as fp:
            obj = json.load(fp)
            if lattice_file is not None:
                obj['lattice']['lattice_file'] = lattice_file
            return cls.model_validate(obj)

    def to_json(self, json_filename: str) -> None:
        """
        Save the SimulatedCommissioning instance to a file in a JSON format.
        """
        with open(json_filename, 'w') as fp:
            obj = self.model_dump()
            json.dump(obj, fp, indent=2)

    def propagate_parents(self) -> None:
        self.magnet_settings._parent = self
        self.design_magnet_settings._parent = self
        self.support_system._parent = self
        self.bpm_system._parent = self

        for rf_settings in [self.rf_settings, self.design_rf_settings]:
            rf_settings._parent = self
            for system_name in rf_settings.systems:
                system = rf_settings.systems[system_name]
                system._parent = rf_settings
                for cav_name in system.cavities:
                    rf_settings.cavities[cav_name]._parent_system = system

        self.injection._parent = self
        self.tuning._parent = self
        self.tuning.tune._parent = self.tuning
        self.tuning.chromaticity._parent = self.tuning
        self.tuning.c_minus._parent = self.tuning
        self.tuning.rf._parent = self.tuning
        return

    def start_server(self, port : int = 13131, timeout: int = 1) -> None:
        _start_server(self, port=port, refresh_rate=timeout)

    def copy(self) -> "SimulatedCommissioning":
        """
        Create a copy of the SimulatedCommissioning instance.
        """
        return SimulatedCommissioning.model_validate(self.model_dump())

    def import_knob(self, json_filename: str) -> None:
        with open(json_filename, 'r') as fp:
            obj = json.load(fp)
            knob_data = KnobData.model_validate(obj)

        for knob_name in knob_data.data.keys():
            assert knob_name not in self.magnet_settings.controls.keys(), f"knob with name {knob_name} already exists SC.magnet_settings."
            assert knob_name not in self.design_magnet_settings.controls.keys(), f"knob with name {knob_name} already exists SC.design_magnet_settings."

        for knob_name in knob_data.data.keys():
            tdata = knob_data.data[knob_name]
            self.magnet_settings.add_knob(knob_name=knob_name, control_names=tdata.control_names, weights=tdata.weights)
            self.design_magnet_settings.add_knob(knob_name=knob_name, control_names=tdata.control_names, weights=tdata.weights)
            logger.info(f'Imported knob {knob_name} with sum(|weights|) = {np.sum(np.abs(tdata.weights)):.2e}')
