'''
Support system: handles all misalignments
'''
import numpy as np
import json
from pydantic import BaseModel
from typing import Optional, Union

from ..utils.sc_tools import update_transformation


class SupportEndpoint(BaseModel):
    """
    Support endpoint: represents an endpoint of a support structure.
    """
    index: int
    supported_by: Optional[tuple[str, int]] = None  # (level, index)
    dx: float = 0.0
    dy: float = 0.0
    s: Optional[float] = None  # s position in the ring, to be filled later


class Support(BaseModel):
    """Support structure: represents a support with two endpoints."""
    start: SupportEndpoint
    end: SupportEndpoint
    supports_elements: list[tuple[str, int]] = []  # list of (level, index) tuples
    length: float = 0.0  # to be filled in add_support
    offset_z: float = 0.0  # not really implemented
    roll: float = 0.0  # only for Level 1
    name: str = 'Support'  # name of the support type, e.g. 'Girder', 'Support', etc.

    @property
    def yaw(self):
        return (self.end.dx - self.start.dx ) / self.length 

    @property
    def pitch(self):
        return (self.end.dy - self.start.dy ) / self.length

    def __repr__(self):
        return f'({self.name}: {self.start.index}-{self.end.index})'


class ElementOffset(BaseModel):
    """
    Element offset: represents an element in the support system with its misalignments.
    """
    index: int
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    roll: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    supported_by: Optional[tuple[str, int]] = None  # (level, index)
    is_bpm: bool = False
    bpm_number: Optional[int] = None  # BPM number if it is a BPM
    s: Optional[float] = None  # s position in the ring, to be filled later


class SupportSystem(BaseModel):
    '''
    Support system: handles all misalignments through a graph-like structure.
    It is composed in level of supports, where L0 is the level of elements (offsets),
    L1 is the level of supports, L2 is the level of supports of supports, etc.
    The structure of the python object is a dictionary of dictionaries, where the keys are the levels (L0, L1, L2, etc.)
    '''
    parent: Optional[object] = None  # Parent object, e.g. the SC object
    data: dict[str, dict[int, Union[ElementOffset, Support]]] = { 'L0' : {} }  # Dictionary to hold the support data, structured by levels

    def add_support(self, index_start, index_end, level=1, name=None):
        assert level >= 1, 'Level must be larger or equal to 1'
        key = f'L{level}'
        if key not in self.data.keys():
            self.data[key] = {}

        if index_start < 0 or index_end < 0:
            raise ValueError('Indices must be non-negative')

        support = Support(start=SupportEndpoint(index=index_start), end=SupportEndpoint(index=index_end), name=name)
        support.start.s = float(self.parent.RING.get_s_pos(index_start)[0])
        support.end.s = float(self.parent.RING.get_s_pos(index_end)[0])

        support.length = (support.end.s - support.start.s) % float(self.parent.RING.circumference)

        index_for_support = len(self.data[key])

        self.data[key][index_for_support] = support

    def add_element(self, index):
        """
        Associates an ElementOffset object (at level L0) to an element of the lattice with index 'index'.
        """
        if index in self.data['L0'].keys():
            raise ValueError(f'Element with index {index} already exists in support system')
        new_element = ElementOffset(index=index)
        if index in self.parent.bpm_system.indices:
            new_element.is_bpm = True
            new_element.bpm_number = self.parent.bpm_system.bpm_number(index)
        new_element.s = float(self.parent.RING.get_s_pos(index)[0])
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
            print(f'Resolving supports: looping through {level}')
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
                    if p_level_key_start is not None or p_level_key_end is not None:
                        if p_level_key_start == p_level_key_end:
                            p_level, p_key = p_level_key_start
                            self.data[p_level][p_key].supports_elements.append((level, key))
                        elif p_level_key_start is not None:
                            p_level, p_key = p_level_key_start
                            self.data[p_level][p_key].supports_elements.append((level, key))
                        elif p_level_key_end is not None:
                            p_level, p_key = p_level_key_end
                            self.data[p_level][p_key].supports_elements.append((level, key))
                        else:
                            raise Exception('Unknown case ?! should not happen.')

        return

    def get_total_offset(self, index, level='L0', endpoint=None):
        if self.level_to_int(level) > 0:
            assert endpoint is not None
            assert endpoint in ['start', 'end'], 'Unknown endpoint type'
        else:
            assert endpoint is None

        if endpoint is None:
            this_element = self.data[level][index]
        elif endpoint == 'start':
            this_element = self.data[level][index].start
        elif endpoint == 'end':
            this_element = self.data[level][index].end
        else:
            raise Exception(f'BUG: Unknown case ?! endpoint={endpoint}')

        off2 = np.array([this_element.dx, this_element.dy])
        p_level_key = this_element.supported_by
        if p_level_key is not None:
            return off2 + self.get_support_offset(this_element.s, p_level_key)
        else:
            return off2

    def get_support_offset(self, s, support_level_key):
        supp_level, supp_index = support_level_key
        support = self.data[supp_level][supp_index]
        s1 = support.start.s
        s2 = support.end.s
        corr_s = 0
        corr_s2 = 0

        ## if support goes through start of ring we need to add corrections
        if support.start.index > support.end.index:
            corr_s2 = self.parent.RING.circumference
            if s < s1:
                corr_s = self.parent.RING.circumference
        ####

        dx1, dy1 = self.get_total_offset(supp_index, supp_level, endpoint='start')
        dx2, dy2 = self.get_total_offset(supp_index, supp_level, endpoint='end')

        dx = (dx2 - dx1)/(s2 - s1 + corr_s2) * (s - s1 + corr_s) + dx1
        dy = (dy2 - dy1)/(s2 - s1 + corr_s2) * (s - s1 + corr_s) + dy1
        return np.array([dx, dy])

    def get_total_rotation(self, index, level='L0'):
        """
        Get the total rotation for an element.
        Returns a tuple (roll, yaw, pitch).
        """
        if self.level_to_int(level) > 0:
            raise NotImplementedError('Total rotation for supports is not implemented yet') 
        eo = self.data[level][index]
        if eo.supported_by is not None:
            support_level, support_key = eo.supported_by
            support = self.data[support_level][support_key]
            yaw = support.yaw + eo.yaw
            pitch = support.pitch + eo.pitch
            roll = support.roll + eo.roll
        else:
            yaw = eo.yaw
            pitch = eo.pitch
            roll = eo.roll

        return roll, pitch, yaw

    def set_offset(self, index, level='L0', endpoint=None, dx=0, dy=0):
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
        elif endpoint == 'start':
            self.data[level][index].start.dx = dx
            self.data[level][index].start.dy = dy
        elif endpoint == 'end':
            self.data[level][index].end.dx = dx
            self.data[level][index].end.dy = dy
        else:
            raise Exception(f'BUG: Unknown case ?! endpoint={endpoint}')

        self.trigger_update(level, index)
        return


    def trigger_update(self, level, index):
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
            dx, dy = self.get_total_offset(eo.index, level)
            dz = eo.dz
            roll, pitch, yaw = self.get_total_rotation(eo.index, level) 

            if eo.is_bpm:
                self.parent.bpm_system.total_offsets_x[eo.bpm_number] = dx
                self.parent.bpm_system.total_offsets_y[eo.bpm_number] = dy
                self.parent.bpm_system.total_rolls[eo.bpm_number] = roll
            else:
                update_transformation(element=self.parent.RING[eo.index],
                                      dx=dx, dy=dy, dz=dz,
                                      roll=roll, yaw=yaw, pitch=pitch)

    def __repr__(self): # hide elements
        return {key: self.data[key] for key in self.data.keys() if key != 'L0'}.__repr__()

    def to_dict(self):
        return self.model_dump(exclude='parent')

    @classmethod
    def from_dict(cls, data, parent=None):
        return cls.model_validate(data, context={'parent': parent})

    def to_json(self, filename):
        with open(filename, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=2)
