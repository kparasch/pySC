from __future__ import annotations
from typing import Literal, Optional, Union, Any, TYPE_CHECKING
from pydantic import BaseModel, model_validator, PrivateAttr, PositiveInt, NonNegativeInt
import logging
from .control import Control, LinearConv
from .multipolar_imperfections import ImperfectionsModel
from .lattice import ATLattice
from .xsuite_lattice import XSuiteLattice

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

MAGNET_NAME_TYPE = Union[str, int]

logger = logging.getLogger(__name__)

class ControlMagnetLink(BaseModel, extra="forbid"):
    link_name: str
    magnet_name: MAGNET_NAME_TYPE
    control_name: str
    component: Literal["A", "B"]
    order: PositiveInt  # index of A/B, starts at 1
    error: LinearConv = LinearConv()
    is_integrated: bool = False

    def value(self, setpoint: float) -> float:
        return self.error.transform(setpoint)

class Magnet(BaseModel, extra="forbid"):
    name: Optional[MAGNET_NAME_TYPE] = None
    sim_index: Optional[NonNegativeInt] = None
    max_order: NonNegativeInt
    A: Optional[list[float]] = None
    B: Optional[list[float]] = None
    offset_A: Optional[list[float]] = None
    offset_B: Optional[list[float]] = None
    to_design: bool = False
    length: Optional[float] = None
    imperfections: Optional[ImperfectionsModel] = None
    _links: list[ControlMagnetLink] = PrivateAttr(default=[])
    _parent = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def initialize_arrays(cls, data: Any) -> Any:
        if "name" not in data or data["name"] is None:
            data["name"] = data["sim_index"]

        max_order = data.get("max_order")

        length = (max_order + 1) if max_order is not None else 0

        keys = ["A", "B", "offset_A", "offset_B"]
        for key in keys:
            if key not in data or data[key] is None:
                data[key] = [0.0] * length

        return data

    @property
    def controls(self) -> list[Control]:
        """
        Returns a list of controls linked to this magnet.
        If the magnet has a parent (e.g., in a settings context), it will return the controls from the parent.
        If no parent is set, it returns the control names from the links.
        """
        if not self._parent:
            raise Exception(
                "Magnet has no settings set in ._parent. Cannot access controls without a parent."
            )
        return [self._parent.controls[link.control_name] for link in self._links]

    @property
    def state(self):
        ## TODO: offset is not treated here. Is this function even useful?
        magnet_name = self.name
        print(f"Magnet: {magnet_name}, max order: {self.max_order}, length: {self.length} m")
        for component, ktype in zip(["B", "A"], ["kn", "ks"]):
            for order in range(self.max_order + 1):
                temp_links = []
                for link in self._links:
                    if link.component == component and link.order - 1 == order:
                        temp_links.append(link)
                if temp_links:
                    actual_value = getattr(self, component)[order]
                    print(
                        f"  {component}{order+1}: {ktype}{order} = {actual_value}, affected by:"
                    )
                    for link in temp_links:
                        control_name = link.control_name
                        setpoint = self._parent.controls[control_name].setpoint
                        link_value = link.value(setpoint)
                        if link.is_integrated:
                            link_value = link_value / self.length
                            # Mirror the order-1 B (dipole/corrector) sign flip
                            # applied in update(), so this diagnostic agrees with
                            # the stored PolynomB[0] value printed above.
                            if link.component == "B" and link.order == 1:
                                link_value = -link_value
                        error = link.error
                        print(
                            f"    - {link.control_name}: setpoint = {setpoint}, error = {repr(error)} -> {link_value}"
                        )

    def update(self):
        # reset components A and B
        self.A = self.offset_A.copy()
        self.B = self.offset_B.copy()

        for link in self._links:
            control = self._parent.controls[link.control_name]
            setpoint = control.setpoint
            value = link.value(setpoint)
            if link.is_integrated:
                assert self.length is not None, f'ERROR: magnet length not specified for integrated strength link: {repr(link)}'
                value = value / self.length
                # AT sign convention for the dipole/corrector term: Δx' = -PolynomB[0]*L,
                # so a positive horizontal kick requires negative PolynomB[0].  Negate the
                # integrated B-component *only* for the order-1 term (PolynomB[0]) so that a
                # positive corrector setpoint produces a positive physical kick, matching
                # MATLAB's SCsetCMs2SetPoints (normBy = [-1, 1] * Length).  Higher-order
                # integrated B strengths (e.g. an integrated quadrupole B2L -> PolynomB[1])
                # must NOT be flipped -- that would invert focusing.
                if link.component == "B" and link.order == 1:
                    value = -value
            if link.component == "A":
                self.A[link.order - 1] += value
            elif link.component == "B":
                self.B[link.order - 1] += value
            else:
                raise ValueError(
                    f"Invalid component '{link.component}' for magnet '{self.name}'"
                )

        if self.imperfections is not None:
            SC: "SimulatedCommissioning" = self._parent._parent
            if isinstance(SC.lattice, ATLattice):
                convention = 'at'
            elif isinstance(SC.lattice, XSuiteLattice):
                convention = 'xsuite'
            else:
                raise ValueError(f"Unknown lattice type {type(SC.lattice)}.")
            Brho = SC.lattice.get_Brho()
            self.B, self.A = self.imperfections.apply(self.B, self.A, Brho=Brho, convention=convention)

        for ii in range(self.max_order + 1):
            self._parent._parent.lattice.set_magnet_component(
                self.sim_index, self.A[ii], 'A', ii, use_design=self.to_design)
            self._parent._parent.lattice.set_magnet_component(
                self.sim_index, self.B[ii], 'B', ii, use_design=self.to_design)

    def ensure_max_order(self, required_max_order: int):
        while self.max_order < required_max_order:
            self.max_order += 1
            self.A.append(0.0)
            self.B.append(0.0)
            self.offset_A.append(0.0)
            self.offset_B.append(0.0)
        SC: "SimulatedCommissioning" = self._parent._parent
        SC.lattice.ensure_max_order(self.sim_index, required_max_order, self.to_design)
