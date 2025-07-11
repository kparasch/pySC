from typing import Dict, Optional
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from .magnet import Magnet, ControlMagnetLink, MAGNET_NAME_TYPE
from .control import Control


class MagnetSettings(BaseModel, extra="forbid"):
    magnets: Dict[MAGNET_NAME_TYPE, Magnet] = Field(default_factory=dict)
    controls: Dict[str, Control] = Field(default_factory=dict)
    links: Dict[str, ControlMagnetLink] = Field(default_factory=dict)
    _ring = PrivateAttr(default=None)

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

    def validate_components_and_get_max_order(self, components: list[str], magnet_name : Optional[MAGNET_NAME_TYPE] = None) -> int:
        """
        Validate the components and return the maximum order needed.
        Each component must be in the form 'A1', 'B2', etc.
        """
        max_order = 0
        for component in components:
            if component[0] not in ["A", "B"]:
                raise ValueError(
                    f"Invalid component '{component}' for magnet '{magnet_name}', must start with 'A' or 'B'"
                )
            if len(component) != 2 or not component[1:].isdigit():
                raise ValueError(
                    f"Invalid component '{component}' for magnet '{magnet_name}', must be in the form 'A1', 'B2', etc."
                )
            order = int(component[1:]) - 1  # Extract the order part
            if order < 0:
                raise ValueError(
                    f"Invalid order in component '{component}' for magnet '{magnet_name}', must be a positive integer"
                )
            if max_order < order:
                max_order = order

        return max_order

    def add_individually_powered_magnet(self,
                                        sim_index: int,
                                        controlled_components: list[str],
                                        magnet_name: str = None) -> None:
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
                        max_order=max_order)
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
            link = ControlMagnetLink(
                link_name=link_name,
                magnet_name=magnet.name,
                control_name=control_name,
                component=component[0],
                order=int(component[1:]),
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

    def set(self, control_name: str, setpoint: float) -> None:
        """
        Set the setpoint for a control by its name.
        This will also update the linked magnets' state.
        """
        if control_name not in self.controls:
            raise ValueError(f"Control '{control_name}' not found")

        control = self.controls[control_name]
        control.check_limits(setpoint)
        control.setpoint = setpoint

        # Update the state of the linked magnets
        magnets = set(link.magnet_name for link in control._links)
        for magnet_name in magnets:
            self.magnets[magnet_name].update()

    def get(self, control_name: str) -> float:
        """
        Get the setpoint for a control by its name.
        """
        if control_name not in self.controls:
            raise ValueError(f"Control '{control_name}' not found")
        return self.controls[control_name].setpoint

    def sendall(self) -> None:
        """
        Send all setpoints to the linked magnets.
        """
        for magnet in self.magnets.values():
            magnet.update()
