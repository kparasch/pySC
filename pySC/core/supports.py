'''
Support system: handles all misalignments
'''
import numpy as np
import json
from pydantic import BaseModel, PrivateAttr
from typing import Optional, Union, TYPE_CHECKING
from pathlib import Path
import logging
from .transformations import (
    as_rotation,
    at_angles_from_rotation,
    at_rotation,
    axis_angle_rotation,
    rotation_from_vectors,
)

if TYPE_CHECKING:
    from .simulated_commissioning import SimulatedCommissioning

logger = logging.getLogger(__name__)
EPS = 1e-12

class ElementOffset(BaseModel, extra="forbid"):
    """
    Element offset: represents an element in the support system with its misalignments.
    """
    index: int
    dx: float = 0.0
    dy: float = 0.0
    ds: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    supported_by: Optional[tuple[str, int]] = None  # (level, index)
    is_bpm: bool = False
    bpm_number: Optional[int] = None  # BPM number if it is a BPM
    s: Optional[float] = None  # center s position in the ring, to be filled later


class SupportEndpoint(BaseModel, extra="forbid"):
    """
    Support endpoint: represents an endpoint of a support structure.
    """
    index: int
    supported_by: Optional[tuple[str, int]] = None  # (level, index)
    dx: float = 0.0
    dy: float = 0.0
    ds: float = 0.0
    s: Optional[float] = None  # center s position in the ring, to be filled later


class Support(BaseModel, extra="forbid"):
    """Support structure: represents a support with two endpoints."""
    start: SupportEndpoint
    end: SupportEndpoint
    supports_elements: list[tuple[str, int]] = []  # list of (level, index) tuples
    length: float = 0.0  # to be filled in add_support
    roll: float = 0.0
    rigid: bool = False
    name: Optional[str] = None  # name of the support type, e.g. 'Girder', 'Support', etc.

    def __repr__(self):
        return f'({self.name}: {self.start.index}-{self.end.index})'


class SupportSystem(BaseModel, extra="forbid"):
    '''
    Support system: handles all misalignments through a graph-like structure.
    It is composed in level of supports, where L0 is the level of elements (offsets),
    L1 is the level of supports, L2 is the level of supports of supports, etc.
    The structure of the python object is a dictionary of dictionaries, where the keys are the levels (L0, L1, L2, etc.)
    '''
    _parent: Optional["SimulatedCommissioning"] = PrivateAttr(default=None)  # Parent object, e.g. the SC object
    data: dict[str, dict[int, Union[ElementOffset, Support]]] = { 'L0' : {} }  # Dictionary to hold the support data, structured by levels
    _reference_X: list[float] = PrivateAttr(default=[])
    _reference_Y: list[float] = PrivateAttr(default=[])
    _reference_Angle: list[float] = PrivateAttr(default=[])

    def initialize_reference_orbit(self) -> None:
        SC = self._parent
        x, y, angle = SC.lattice.get_reference_orbit()
        self._reference_X = x
        self._reference_Y = y
        self._reference_Angle = angle
        return
    
    def add_support(self, index_start, index_end, level=1, name=None):
        assert level >= 1, 'Level must be larger or equal to 1'
        logger.debug(f'Adding support with {index_start=}, {index_end=} in {level=}')
        key = f'L{level}'
        if key not in self.data.keys():
            self.data[key] = {}

        if index_start < 0 or index_end < 0:
            raise ValueError('Indices must be non-negative')

        if name is None:
            name = 'Support'

        support = Support(start=SupportEndpoint(index=index_start), end=SupportEndpoint(index=index_end), name=name)
        twiss_s = self._twiss_s()
        support.start.s = self._element_center_s(index_start)
        support.end.s = self._element_center_s(index_end)

        support.length = (support.end.s - support.start.s) % float(twiss_s[-1])

        index_for_support = len(self.data[key])

        self.data[key][index_for_support] = support

        return index_for_support

    def add_element(self, index):
        """
        Associates an ElementOffset object (at level L0) to an element of the lattice with index 'index'.
        """
        if index in self.data['L0'].keys():
            raise ValueError(f'Element with index {index} already exists in support system')
        new_element = ElementOffset(index=index)
        if hasattr(self._parent, 'bpm_system') and index in self._parent.bpm_system.indices:
            new_element.is_bpm = True
            new_element.bpm_number = self._parent.bpm_system.bpm_number(index=index)
        new_element.s = self._element_center_s(index)
        self.data['L0'][int(index)] = new_element

    def look_for_support(self, my_level, my_index):
        """
        Look for the first support structure that supports the element at my_index
        in the level my_level.
        Returns a tuple (next_level, support_key) if found, otherwise None.
        """
        all_levels = list(self.data.keys())

        # remove levels below, and my level, from list of levels to look into 
        # i.e. keep only higher levels
        int_level = int(my_level[1:])
        for ii in range(0, int_level+1):
            all_levels.remove(f'L{ii}')

        # Loop through all next levels until we find the first supporting structure
        for next_level in all_levels:
            for support_key in self.data[next_level].keys():
                support = self.data[next_level][support_key]
                if support.start.index < support.end.index:
                    ## normal support
                    if my_index >= support.start.index and my_index <= support.end.index:
                        return (next_level, support_key)
                else:
                    ## support passes through start of lattice
                    if my_index >= support.start.index or my_index <= support.end.index:
                        return (next_level, support_key)
        return None

    def check_levels_are_sorted(self):
        int_levels = [int(str(key)[1:]) for key in self.data.keys()]
        return all(int_levels[i] < int_levels[i+1] for i in range(len(int_levels)-1))

    def level_to_int(self, level):
        return int(level[1:])

    def sorted_levels(self):
        int_levels = sorted([self.level_to_int(level) for level in self.data.keys()])
        return [f'L{level}' for level in int_levels]

    def resolve_graph(self):
        """
        Resolve the support graph by finding which elements are supported by which supports.
        This will populate the `supported_by` attribute of each element and endpoint,
        and the `supports_elements` attribute of each support.
        """
        all_levels = self.sorted_levels()
        assert self.check_levels_are_sorted(), 'BUG: why are levels not sorted ?!'

        ## for each element/endpoint find who it is supported by
        for level in all_levels:
            logger.info(f'Resolving supports: looping through {level}')
            for key in self.data[level].keys():
                if level == 'L0':
                    # element offset
                    index = self.data[level][key].index
                    self.data[level][key].supported_by = self.look_for_support(level, index)
                else: # level > 0
                    # start endpoint
                    index_start = self.data[level][key].start.index
                    self.data[level][key].start.supported_by = self.look_for_support(level, index_start)
                    # end endpoint
                    index_end = self.data[level][key].end.index
                    self.data[level][key].end.supported_by = self.look_for_support(level, index_end)

        ## populate who supports who based on who is supported by who
        for level in all_levels:
            for key in self.data[level].keys():
                if level == 'L0':
                    p_level_key = self.data[level][key].supported_by
                    if p_level_key is not None:
                        p_level, p_key = p_level_key
                        self.data[p_level][p_key].supports_elements.append((level, key))
                else: ## level > 0, go per endpoint
                    p_level_key_start = self.data[level][key].start.supported_by
                    p_level_key_end = self.data[level][key].end.supported_by
                    parent_keys = {p_level_key_start, p_level_key_end}
                    parent_keys.discard(None)
                    for p_level, p_key in parent_keys:
                        self.data[p_level][p_key].supports_elements.append((level, key))

        return

    def _ensure_reference_orbit(self) -> None:
        if len(self._reference_X) == 0 or len(self._reference_Y) == 0 or len(self._reference_Angle) == 0:
            self.initialize_reference_orbit()

    def _twiss_s(self) -> np.ndarray:
        return np.asarray(self._parent.lattice.twiss['s'], dtype=float)

    def _element_center_s(self, index: int) -> float:
        """
        Return the longitudinal center of an element from entrance/exit refpoints.
        """
        s_ref = self._twiss_s()
        index = int(index)
        if index < 0:
            raise ValueError('Indices must be non-negative')
        if index + 1 >= len(s_ref):
            raise IndexError(
                f'Element index {index} has no exit refpoint in lattice.twiss["s"]. '
                'SupportSystem needs entrance and exit refpoints to compute element centers.'
            )

        entrance = float(s_ref[index])
        exit_ = float(s_ref[index + 1])
        circumference = float(s_ref[-1])
        if exit_ < entrance and circumference > 0:
            exit_ += circumference
        center = 0.5 * (entrance + exit_)
        if circumference > 0:
            center = center % circumference
        return center

    def _reference_pose(self, index_or_s):
        """
        Return the design world pose at an element center index or longitudinal s.
        The returned rotation maps local [dx, dy, ds] to world [X, Y, Z].
        """
        self._ensure_reference_orbit()
        x_ref = np.asarray(self._reference_X, dtype=float)
        y_ref = np.asarray(self._reference_Y, dtype=float)
        angle_ref = np.asarray(self._reference_Angle, dtype=float)

        if isinstance(index_or_s, (int, np.integer)):
            s = self._element_center_s(int(index_or_s))
        else:
            s = float(index_or_s)

        s_ref = self._twiss_s()
        circumference = float(s_ref[-1])
        if circumference > 0:
            s = s % circumference
            if np.isclose(s, 0.0) and float(index_or_s) > 0:
                s = circumference
        theta_ref = np.unwrap(angle_ref)
        x = np.interp(s, s_ref, x_ref)
        y = np.interp(s, s_ref, y_ref)
        theta = np.interp(s, s_ref, theta_ref)

        p = np.array([x, y, 0.0])
        R = np.array([[-np.sin(theta), 0.0, np.cos(theta)],
                      [ np.cos(theta), 0.0, np.sin(theta)],
                      [           0.0, 1.0,           0.0]])
        return p, R

    def _parent_pose_at_s(self, s, parent_key):
        if parent_key is None:
            return self._reference_pose(float(s))
        return self._support_pose_at_s(float(s), parent_key)

    def _endpoint_parent_pose(self, endpoint):
        if endpoint.supported_by is None:
            return self._reference_pose(endpoint.index)
        return self._support_pose_at_s(endpoint.s, endpoint.supported_by)

    def _support_fraction(self, s, support):
        s1 = support.start.s
        s2 = support.end.s
        corr_s = 0.0
        corr_s2 = 0.0
        circumference = float(self._twiss_s()[-1])
        if support.start.index > support.end.index:
            corr_s2 = circumference
            if s < s1:
                corr_s = circumference
        denominator = s2 - s1 + corr_s2
        if abs(denominator) < EPS:
            return 0.0
        return (s - s1 + corr_s) / denominator

    @staticmethod
    def _support_design_frame(design_chord, fallback_R):
        z_norm = np.linalg.norm(design_chord)
        z_axis = design_chord / z_norm if z_norm >= EPS else fallback_R[:, 2]

        y_axis = np.array([0.0, 0.0, 1.0])
        y_axis = y_axis - np.dot(y_axis, z_axis) * z_axis
        if np.linalg.norm(y_axis) < EPS:
            y_axis = fallback_R[:, 1] - np.dot(fallback_R[:, 1], z_axis) * z_axis
        if np.linalg.norm(y_axis) < EPS:
            y_axis = np.array([0.0, 1.0, 0.0]) - np.dot(np.array([0.0, 1.0, 0.0]), z_axis) * z_axis

        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _support_endpoint_positions(self, support_level_key):
        supp_level, supp_index = support_level_key
        support = self.data[supp_level][supp_index]

        design_start, _ = self._reference_pose(support.start.index)
        design_end, _ = self._reference_pose(support.end.index)
        base_start, R_start_parent = self._endpoint_parent_pose(support.start)
        base_end, R_end_parent = self._endpoint_parent_pose(support.end)

        start_offset = np.array([support.start.dx, support.start.dy, support.start.ds])
        end_offset = np.array([support.end.dx, support.end.dy, support.end.ds])
        start = base_start + R_start_parent @ start_offset
        end = base_end + R_end_parent @ end_offset

        if not support.rigid:
            return design_start, design_end, start, end

        nominal_length = np.linalg.norm(design_end - design_start)
        new_length = np.linalg.norm(end - start)
        if nominal_length < EPS or new_length < EPS:
            return design_start, design_end, start, end

        scale = nominal_length / new_length
        center = 0.5 * (start + end)
        start = center + scale * (start - center)
        end = center + scale * (end - center)
        return design_start, design_end, start, end

    def _support_pose_at_s(self, s, support_level_key):
        supp_level, supp_index = support_level_key
        support = self.data[supp_level][supp_index]
        p_ref, R_ref = self._reference_pose(float(s))
        design_start, design_end, start, end = self._support_endpoint_positions(support_level_key)
        fraction = self._support_fraction(float(s), support)
        displacement_start = start - design_start
        displacement_end = end - design_end
        displacement = displacement_start + fraction * (displacement_end - displacement_start)
        p = p_ref + displacement

        design_chord = design_end - design_start
        corrected_chord = end - start
        R_design = self._support_design_frame(design_chord, R_ref)
        R_align = rotation_from_vectors(design_chord, corrected_chord).as_matrix()
        axis_norm = np.linalg.norm(corrected_chord)
        if axis_norm < EPS:
            roll_axis = R_align @ R_design[:, 2]
        else:
            roll_axis = corrected_chord / axis_norm
        R_roll = axis_angle_rotation(roll_axis, support.roll).as_matrix()
        R = R_roll @ R_align @ R_design
        return p, R

    def _element_pose(self, index):
        eo = self.data['L0'][index]
        if eo.supported_by is None:
            parent_p, parent_R = self._reference_pose(eo.index)
        else:
            parent_p, parent_R = self._support_pose_at_s(eo.s, eo.supported_by)

        offset = np.array([eo.dx, eo.dy, eo.ds])
        R_local = at_rotation(pitch=eo.pitch, yaw=eo.yaw, roll=eo.roll).as_matrix()
        p = parent_p + parent_R @ offset
        R = parent_R @ R_local
        return p, R

    def get_total_offset(self, index, level='L0', endpoint=None):
        if self.level_to_int(level) > 0:
            assert endpoint is not None
            assert endpoint in ['start', 'end'], 'Unknown endpoint type'
        else:
            assert endpoint is None

        if endpoint is None:
            p_world, _ = self._element_pose(index)
            p_ref, R_ref = self._reference_pose(self.data[level][index].index)
        elif endpoint == 'start':
            p_ref, R_ref = self._reference_pose(self.data[level][index].start.index)
            _, _, start, _ = self._support_endpoint_positions((level, index))
            p_world = start
        elif endpoint == 'end':
            p_ref, R_ref = self._reference_pose(self.data[level][index].end.index)
            _, _, _, end = self._support_endpoint_positions((level, index))
            p_world = end
        else:
            raise Exception(f'BUG: Unknown case ?! endpoint={endpoint}')

        return R_ref.T @ (p_world - p_ref)

    def get_support_offset(self, s, support_level_key):
        p_world, _ = self._support_pose_at_s(float(s), support_level_key)
        p_ref, R_ref = self._reference_pose(float(s))
        return R_ref.T @ (p_world - p_ref)

    def get_total_rotation(self, index, level='L0'):
        """
        Get the total rotation for an element.
        Returns a SciPy Rotation in the local design frame.
        """
        if self.level_to_int(level) > 0:
            raise NotImplementedError('Total rotation for supports is not implemented yet') 
        _, R_world = self._element_pose(index)
        _, R_ref = self._reference_pose(self.data[level][index].index)
        R_total = R_ref.T @ R_world
        return as_rotation(R_total)

    def set_offset(self, index, level='L0', endpoint=None, dx=0, dy=0, ds=0):
        """
        Set the transverse offset for an element or endpoint.
        """
        if self.level_to_int(level) > 0:
            assert endpoint is not None
            assert endpoint in ['start', 'end'], 'Unknown endpoint type'
        else:
            assert endpoint is None

        if endpoint is None:
            self.data[level][index].dx = dx
            self.data[level][index].dy = dy
            self.data[level][index].ds = ds
        elif endpoint == 'start':
            self.data[level][index].start.dx = dx
            self.data[level][index].start.dy = dy
            self.data[level][index].start.ds = ds
        elif endpoint == 'end':
            self.data[level][index].end.dx = dx
            self.data[level][index].end.dy = dy
            self.data[level][index].end.ds = ds
        else:
            raise Exception(f'BUG: Unknown case ?! endpoint={endpoint}')

        self.trigger_update(level, index)
        return

    def trigger_update(self, level: str, index):
        """
        Trigger the update of the transformations for the given level and index.
        If the target is a support, it will trigger the transformation of the elements it supports.
        If the target is an element, it will recompute the at matrices T1,T2,R1,R2.
        """
        if level != 'L0':
            assert isinstance(self.data[level][index], Support), f'Element {index} in level {level} is not a Support object'
            for trig_level, trig_index in self.data[level][index].supports_elements:
                self.trigger_update(trig_level, trig_index)
        else:
            eo = self.data[level][index]
            dx, dy, ds = self.get_total_offset(eo.index, level)
            rot = self.get_total_rotation(eo.index, level)

            if eo.is_bpm:
                roll, _, _ = at_angles_from_rotation(rot)
                self._parent.bpm_system.offsets_x[eo.bpm_number] = dx
                self._parent.bpm_system.offsets_y[eo.bpm_number] = dy
                self._parent.bpm_system.rolls[eo.bpm_number] = roll
                self._parent.bpm_system.update_rot_matrices()
            else:
                self._parent.lattice.update_misalignment(index=eo.index, dx=dx, dy=dy, ds=ds, rot=rot)

    def update_all(self) -> None:
        for index in self.data['L0'].keys():
            self.trigger_update('L0', index)
        return

    ## this should maybe belong to tuning algorithms
    def fake_align_bpms(self, bpm_indices, magnet_indices):
        logger.fatal('Function is deprecated, use the one from SC.tuning instead.')
        for bpm_index, magnet_index in zip(bpm_indices, magnet_indices):
            magnet_dx, magnet_dy = self.get_total_offset(index=magnet_index)[:2]
            bpm_tot_dx, bpm_tot_dy = self.get_total_offset(index=bpm_index)[:2]
            new_dx = magnet_dx - bpm_tot_dx
            new_dy = magnet_dy - bpm_tot_dy
            bpm_number = self._parent.bpm_system.bpm_number(index=bpm_index)
            self._parent.bpm_system.bba_offsets_x[bpm_number] = new_dx
            self._parent.bpm_system.bba_offsets_y[bpm_number] = new_dy

#     def __repr__(self): # hide elements
#         return {key: self.data[key] for key in self.data.keys() if key != 'L0'}.__repr__()

    def to_dict(self):
        return self.model_dump(exclude='parent')

    @classmethod
    def from_dict(cls, data, parent=None):
        return cls.model_validate(data, context={'parent': parent})

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=2)

    @classmethod
    def from_json(cls, filename, parent=None):
        json_string = Path(filename).read_text()
        return cls.model_validate_json(json_string, context={'parent': parent})
