from typing import Dict, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from .magnet import Magnet, ControlMagnetLink, MAGNET_NAME_TYPE
from .control import Control

if TYPE_CHECKING:
    from .new_simulated_commissioning import SimulatedCommissioning

class MagnetSettings(BaseModel, extra="forbid"):
    magnets: Dict[MAGNET_NAME_TYPE, Magnet] = Field(default_factory=dict)
    controls: Dict[str, Control] = Field(default_factory=dict)
    links: Dict[str, ControlMagnetLink] = Field(default_factory=dict)
    index_mapping: Dict[int, MAGNET_NAME_TYPE] = Field(default_factory=dict)
    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def check_links_references(self):
        magnet_names = set(self.magnets.keys())
        control_names = set(self.controls.keys())

        for link_key, link in self.links.items():
            if link.magnet_name not in magnet_names:
                raise ValueError(
                    f"Link '{link_key}' references unknown magnet '{link.magnet_name}'"
                )
            if link.control_name not in control_names:
                raise ValueError(
                    f"Link '{link_key}' references unknown control '{link.control_name}'"
                )

        for magnet in self.magnets.values():
            magnet._parent = self  # Set the parent to the current settings instance
        self.connect_links()
        return self

    def add_magnet(self, magnet: Magnet) -> None:
        if magnet.name in self.magnets:
            raise ValueError(f"Magnet '{magnet.name}' already exists")
        self.magnets[magnet.name] = magnet
        self.index_mapping[magnet.sim_index] = magnet.name

    def add_control(self, control: Control) -> None:

        ## TODO allow to change name if it is already used

        if control.name in self.controls:
            raise ValueError(f"Control '{control.name}' already exists")
        self.controls[control.name] = control

    def add_link(self, link: ControlMagnetLink) -> None:
        """Add a link between a control and a magnet."""
        if link.magnet_name not in self.magnets:
            raise ValueError(f"Magnet '{link.magnet_name}' not found")
        if link.control_name not in self.controls:
            raise ValueError(f"Control '{link.control_name}' not found")
        self.links[link.link_name] = link

    def validate_one_component(self, component: str, magnet_name: Optional[MAGNET_NAME_TYPE] = None) -> tuple[str, int]:
        if component[0] not in ["A", "B"]:
            raise ValueError(
                f"Invalid component '{component}' for magnet '{magnet_name}', must start with 'A' or 'B'"
            )
        if component[-1] == 'L':
            digit_str = component[1:-1]
        else:
            digit_str = component[1:]

        if not digit_str.isdigit():
            raise ValueError(
                f"Invalid component '{component}' for magnet '{magnet_name}', must be in the form 'A1', 'B2', etc."
            )
        order = int(digit_str) - 1  # Extract the order part
        if order < 0:
            raise ValueError(
                f"Invalid order in component '{component}' for magnet '{magnet_name}', must be a positive integer"
            )
        return component[0], order

    def validate_components_and_get_max_order(self, components: list[str], magnet_name : Optional[MAGNET_NAME_TYPE] = None) -> int:
        """
        Validate the components and return the maximum order needed.
        Each component must be in the form 'A1', 'B2', etc.
        """
        max_order = 0
        for component in components:
            _, order = self.validate_one_component(component, magnet_name=magnet_name)
            if max_order < order:
                max_order = order

        return max_order

    def add_individually_powered_magnet(self,
                                        sim_index: int,
                                        controlled_components: list[str],
                                        magnet_name: Optional[str] = None,
                                        magnet_length: Optional[float] = None,
                                        to_design: bool = False) -> None:
        """
        Add a magnet with individually powered components.
        Each component must be controlled by a separate control.
        """
        if magnet_name is not None and magnet_name in self.magnets:
            raise ValueError(f"Magnet '{magnet_name}' already exists")

        if magnet_name is None:
            magnet_name = sim_index  # Use sim_index as the default name if not provided

        max_order = self.validate_components_and_get_max_order(components=controlled_components, magnet_name=magnet_name)

        # Create a new Magnet instance with the specified components
        magnet = Magnet(name=magnet_name,
                        sim_index=sim_index,
                        max_order=max_order,
                        to_design=to_design,
                        length=magnet_length)
        magnet._parent = self  # Set the parent to the current settings instance
        self.add_magnet(magnet)

        # Create controls for each component
        for component in controlled_components:
            control_name = f"{magnet.name}/{component}"
            control = Control(name=control_name, setpoint=0.0)
            self.add_control(control)

        # Create links for each component
        for component in controlled_components:
            control_name = f"{magnet.name}/{component}"
            link_name = f"{control_name}->{control_name}"
            is_integrated = True if component[-1] == 'L' else False
            order = int(component[1:-1]) if is_integrated else int(component[1:])
            link = ControlMagnetLink(
                link_name=link_name,
                magnet_name=magnet.name,
                control_name=control_name,
                component=component[0],
                order=order,
                is_integrated=is_integrated
            )
            self.add_link(link)

    def connect_links(self) -> None:
        # Clear any previous links
        for control in self.controls.values():
            control._links.clear()
        for magnet in self.magnets.values():
            magnet._links.clear()

        # Populate targets and links
        for link in self.links.values():
            self.controls[link.control_name]._links.append(link)
            self.magnets[link.magnet_name]._links.append(link)

    def set(self, control_name: str, setpoint: float, use_design: bool = False) -> None:
        """
        Set the setpoint for a control by its name.
        This will also update the linked magnets' state.
        """
        if use_design: # go do it on the design instead using the same method of the design magnet settings
            self._parent.design_magnet_settings.set(control_name=control_name, setpoint=setpoint)
            return

        if control_name not in self.controls:
            raise ValueError(f"Control '{control_name}' not found")

        control = self.controls[control_name]
        control.check_limits_and_set(setpoint)
        #control.setpoint = setpoint

        # Update the state of the linked magnets
        magnets = set(link.magnet_name for link in control._links)
        for magnet_name in magnets:
            self.magnets[magnet_name].update()

    def get(self, control_name: str, use_design: bool = False) -> float:
        """
        Get the setpoint for a control by its name.
        """
        if use_design: # go do it on the design instead using the same method of the design magnet settings
            return self._parent.design_magnet_settings.get(control_name=control_name)

        if control_name not in self.controls:
            raise ValueError(f"Control '{control_name}' not found")
        return self.controls[control_name].setpoint

    def sendall(self) -> None:
        """
        Send all setpoints to the linked magnets.
        """
        for magnet in self.magnets.values():
            magnet.update()

    def get_many(self, control_list: list[str], use_design: bool = False) -> dict[str, float]:
        data = {}
        for control_name in control_list:
            data[control_name] = self.get(control_name, use_design=use_design)
        return data

    def set_many(self, data: dict[str, float], use_design: bool = False) -> None:
        for control_name in data.keys():
            self.set(control_name, data[control_name], use_design=use_design)
        return