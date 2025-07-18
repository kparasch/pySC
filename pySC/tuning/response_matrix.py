from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Literal
from ..core.numpy_type import NPARRAY
import numpy as np

class InverseResponseMatrix(BaseModel, extra="forbid"):
    matrix: NPARRAY
    method: Literal['tikhonov', 'svd_values', 'svd_cutoff']
    parameter: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def dot(self, output: np.array) -> np.array:
        return np.dot(self.matrix, output)
    
    @property
    def shape(self):
        return self.matrix.shape

class ResponseMatrix(BaseModel, extra="forbid"):
    #inputs -> columns -> axis = 1
    #outputs -> rows -> axis = 0
    # here, good and bad in the names of the variables mean that bad output/input includes inside
    # values which are marked to be ignored (like bad bpms).
    RM: NPARRAY

    # input_indices: list[int]
    # input_names: list[int, str]

    # output_indices: list[int]
    # output_names: list[int, str]

    _n_outputs: int = 0
    _n_inputs: int = 0
    _singular_values: Optional[NPARRAY] = None
    _bad_outputs: list[int] = PrivateAttr(default=[])
    _bad_inputs: list[int] = PrivateAttr(default=[])

    _output_mask: NPARRAY = np.array([])
    _inverse_RM: Optional[InverseResponseMatrix] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def initialize_and_check(self):
        self._n_outputs, self._n_inputs = self.RM.shape
        self._singular_values = np.linalg.svd(self.RM, compute_uv=False)
        self.make_masks()
        return self

    @property
    def singular_values(self) -> np.array:
        return self._singular_values

    @property
    def bad_inputs(self) -> list[int]:
        return self._bad_inputs

    @bad_inputs.setter
    def bad_inputs(self, bad_list: list[int]) -> None:
        self._bad_inputs = bad_list.copy()
        self.make_masks()

    @property
    def good_inputs(self) -> list[int]:
        return self._bad_inputs

    @good_inputs.setter
    def good_inputs(self, good_list: list[int]) -> None:
        self._good_inputs = good_list.copy()
        self.make_masks()

    def make_masks(self):
        self._output_mask = np.ones(self._n_outputs, dtype=bool)
        self._output_mask[self._bad_outputs] = False
        self._input_mask = np.ones(self._n_inputs, dtype=bool)
        self._input_mask[self._bad_inputs] = False

    def build_pseudoinverse(self, method='svd_cutoff', parameter: float = 0.):
        matrix = self.RM[self._output_mask, :][:, self._input_mask]
        U, s_mat, Vh = np.linalg.svd(matrix, full_matrices=False)

        if method == 'svd_cutoff':
            cutoff = parameter
            s0 = s_mat[0]
            s_mat[s_mat < cutoff * s0] = 0
        elif method == 'svd_values':
            number_of_values_to_keep = parameter
            s_mat[int(number_of_values_to_keep):] = 0
        elif method == 'tikhonov':
            alpha = parameter
            s_mat = s_mat / (np.square(s_mat) + alpha**2)

        matrix_inv = np.dot(np.dot(np.transpose(Vh), np.diag(s_mat)), np.transpose(U))

        return InverseResponseMatrix(matrix=matrix_inv, method=method, parameter=parameter)

    def solve(self, output: np.array, method: str = 'svd_cutoff', parameter: float = 0.):
        if self._inverse_RM is None:
            self._inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter)
        else:
            if self._inverse_RM.method != method or self._inverse_RM.parameter != parameter:
                self._inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter)

        bad_output = output.copy()
        bad_output[np.isnan(bad_output)] = 0
        good_output = bad_output[self._output_mask]

        expected_shape = (self._n_inputs - len(self._bad_inputs), len(good_output))
        if self._inverse_RM.shape != expected_shape:
            raise Exception('Error: shapes of Response matrix, excluding bad inputs and outputs do not match: \n' 
             + f'inverse RM shape = {self._inverse_RM.shape},\n'
             + f'expected inputs: {len(self.input_names)} - {len(self._bad_inputs)},\n'
             + f'expected outputs: {len(self.output_names)} - {len(self._bad_outputs)}'
             + f'received outputs: {len(output)}, should be equal to {len(self.output_names)}!')

        good_input = self._inverse_RM.dot(good_output)

        bad_input = np.zeros(self._n_inputs, dtype=float)
        bad_input[self._input_mask] = good_input
        return bad_input

