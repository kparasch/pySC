from __future__ import annotations
from typing import Literal, Optional, Union, Any
from pydantic import BaseModel, model_validator, PrivateAttr

MAGNET_NAME_TYPE = Union[str, int]


class LinearConv(BaseModel, extra="forbid"):
    factor: float = 1.0
    offset: float = 0.0

    def transform(self, value: float) -> float:
        return value * self.factor + self.offset


class Control(BaseModel, extra="forbid"):
    name: str
    setpoint: float
    calibration: LinearConv = LinearConv()
    _links: Optional[list[ControlMagnetLink]] = PrivateAttr(default=[])


class Magnet(BaseModel, extra="forbid"):
    name: Optional[MAGNET_NAME_TYPE] = None
    sim_index: int = None
    max_order: int
    A: Optional[list[float]] = None
    B: Optional[list[float]] = None
    _links: Optional[list[ControlMagnetLink]] = PrivateAttr(default=[])
    _parent = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def initialize_arrays(cls, data: Any) -> Any:
        if "name" not in data or data["name"] is None:
            data["name"] = data["sim_index"]

        max_order = data.get("max_order")

        length = (max_order + 1) if max_order is not None else 0

        if "A" not in data or data["A"] is None:
            data["A"] = [0.0] * length
        if "B" not in data or data["B"] is None:
            data["B"] = [0.0] * length

        return data

    @property
    def controls(self) -> list[Control]:
        """
        Returns a list of controls linked to this magnet.
        If the magnet has a parent (e.g., in a settings context), it will return the controls from the parent.
        If no parent is set, it returns the control names from the links.
        """
        if self._parent and self._links:
            return [self._parent.controls[link.control_name] for link in self._links]
        return [link.control_name for link in self._links] if self._links else []

    @property
    def state(self):
        manget_name = self.name
        print(f"Magnet: {manget_name}, max order: {self.max_order}")
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
                        error = link.conv
                        print(
                            f"    - {link.control_name}: setpoint = {setpoint}, error = {repr(error)} -> {link_value}"
                        )

    def update(self):
        # reset components A and B
        self.A = [0.0] * (self.max_order + 1)
        self.B = [0.0] * (self.max_order + 1)

        for link in self._links:
            control = self._parent.controls[link.control_name]
            setpoint = control.setpoint
            value = link.value(setpoint)
            if link.component == "A":
                self.A[link.order - 1] += value
            elif link.component == "B":
                self.B[link.order - 1] += value
            else:
                raise ValueError(
                    f"Invalid component '{link.component}' for magnet '{self.name}'"
                )


class ControlMagnetLink(BaseModel, extra="forbid"):
    link_name: str
    magnet_name: MAGNET_NAME_TYPE
    control_name: str
    component: Literal["A", "B"]
    order: int  # index of A/B, starts at 1
    conv: LinearConv = LinearConv()

    def value(self, setpoint: float) -> float:
        return self.conv.transform(setpoint)
