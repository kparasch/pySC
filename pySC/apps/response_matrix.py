from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Literal
from ..core.types import NPARRAY
import numpy as np
import logging
import json

PLANE_TYPE = Literal['H', 'V', 'Q', 'SQ']

logger = logging.getLogger(__name__)

class InverseResponseMatrix(BaseModel, extra="forbid"):
    matrix: NPARRAY
    method: Literal['tikhonov', 'svd_values', 'svd_cutoff', 'micado']
    parameter: float
    zerosum: bool = True
    rf: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def dot(self, output: np.array) -> np.array:
        return np.dot(self.matrix, output)

    @property
    def shape(self):
        return self.matrix.shape

class ResponseMatrix(BaseModel):
    #inputs -> columns -> axis = 1
    #outputs -> rows -> axis = 0
    # here, good and bad in the names of the variables mean that bad output/input includes inside
    # values which are marked to be ignored (e.g. bad bpms are bad_outputs).
    matrix: NPARRAY

    input_names: Optional[list[str]] = None
    output_names: Optional[list[str]] = None
    inputs_plane: Optional[list[PLANE_TYPE]] = None
    outputs_plane: Optional[list[PLANE_TYPE]] = None
    rf_response: Optional[NPARRAY] = None
    rf_weight: float = 1

    _n_outputs: int = PrivateAttr(default=0)
    _n_inputs: int = PrivateAttr(default=0)
    _singular_values: Optional[NPARRAY] = PrivateAttr(default=None)
    _bad_outputs: list[int] = PrivateAttr(default=[])
    _bad_inputs: list[int] = PrivateAttr(default=[])

    _output_mask: Optional[NPARRAY] = PrivateAttr(default=None)
    _input_mask: Optional[NPARRAY] = PrivateAttr(default=None)
    _inverse_RM: Optional[InverseResponseMatrix] = PrivateAttr(default=None)
    _inverse_RM_H: Optional[InverseResponseMatrix] = PrivateAttr(default=None)
    _inverse_RM_V: Optional[InverseResponseMatrix] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def RM(self):
        logger.warning('ResponseMatrix.RM is deprecated! Please use ResponseMatrix.matrix instead.')
        return self.matrix

    @model_validator(mode='after')
    def initialize_and_check(self):
        self._n_outputs, self._n_inputs = self.matrix.shape
        try:
            self._singular_values = np.linalg.svd(self.matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            logger.warning('SVD of the response matrix failed, correction will be impossible.')
            self._singular_values = None
        self.make_masks()

        if self.inputs_plane is None:
            Nh = self._n_inputs // 2
            if self._n_inputs % 2 != 0:
                logger.warning('Plane of inputs is undefined and number of inputs in response matrix is not even. '
                               'Misinterpretation of the input plane is guaranteed!')
            self.inputs_plane = ['H'] * Nh + ['V'] * (self._n_inputs - Nh)

        if self.outputs_plane is None:
            Nh = self._n_outputs // 2
            if self._n_outputs % 2 != 0:
                logger.warning('Plane of outputs is undefined and number of outputs in response matrix is not even. '
                               'Misinterpretation of the output plane is guaranteed!')
            self.outputs_plane = ['H'] * Nh + ['V'] * (self._n_outputs - Nh)

        if self.rf_response is None:
            self.rf_response = np.zeros(self._n_outputs)
        else:
            if len(self.rf_response) != self._n_outputs:
                logger.warning(f'RF response does not have the correct length: {len(self.rf_response)} (should have been {self._n_outputs}).')
                logger.warning('RF response will be removed.')
                self.rf_response = np.zeros(self._n_outputs)

        return self

    @property
    def singular_values(self) -> np.array:
        return self._singular_values

    def set_rf_response(self, rf_response: np.array, plane=None) -> None:
        assert plane is None or plane in ['H', 'V'], f"Unknown plane: {plane}."
        len_rf = len(rf_response)
        if plane is None:
            assert len_rf == self._n_outputs, f"Incorrect rf_response length: {len_rf} (instead of {self._n_outputs})."
            self.rf_response = np.array(rf_response)
        else:
            output_plane_mask = self.get_output_plane_mask(plane=plane)
            n_plane = sum(output_plane_mask)
            assert len_rf == n_plane, f"Incorrect rf_response length for plane {plane}: {len_rf} (instead of {n_plane})."
            self.rf_response[output_plane_mask] = np.array(rf_response)
        return

    def get_matrix_in_plane(self, plane: Optional[PLANE_TYPE] = None) -> np.array:
        if plane is None:
            return self.matrix
        else:
            output_plane_mask = self.get_output_plane_mask(plane)
            input_plane_mask = self.get_input_plane_mask(plane)
            return self.matrix[output_plane_mask, :][:, input_plane_mask]

    def get_input_plane_mask(self, plane: Literal[PLANE_TYPE]) -> np.array:
        return np.array(self.inputs_plane) == plane

    def get_output_plane_mask(self, plane: Literal[PLANE_TYPE]) -> np.array:
        return np.array(self.outputs_plane) == plane

    @property
    def matrix_h(self) -> np.array:
        return self.get_matrix_in_plane(plane='H')

    @property
    def matrix_v(self) -> np.array:
        return self.get_matrix_in_plane(plane='V')

    @property
    def bad_inputs(self) -> list[int]:
        return self._bad_inputs

    @bad_inputs.setter
    def bad_inputs(self, bad_list: list[int]) -> None:
        if self._bad_inputs != bad_list:
            self._bad_inputs = bad_list.copy()
            self.make_masks()

    @property
    def bad_outputs(self) -> list[int]:
        return self._bad_outputs

    @bad_outputs.setter
    def bad_outputs(self, bad_list: list[int]) -> None:
        if self._bad_outputs != bad_list:
            self._bad_outputs = bad_list.copy()
            self.make_masks()

    def make_masks(self):
        self._inverse_RM = None # discard inverse RM, by changing bad inputs/outputs it becomes invalid
        self._inverse_RM_H = None # discard inverse RM, by changing bad inputs/outputs it becomes invalid
        self._inverse_RM_V = None # discard inverse RM, by changing bad inputs/outputs it becomes invalid
        self._output_mask = np.ones(self._n_outputs, dtype=bool)
        self._output_mask[self._bad_outputs] = False
        self._input_mask = np.ones(self._n_inputs, dtype=bool)
        self._input_mask[self._bad_inputs] = False

    def disable_inputs(self, inputs: list[str]):
        assert self.input_names is not None, "ResponseMatrix.input_names are not defined"
        for _input in inputs:
            assert _input in self.input_names, f"{_input} not found in ResponseMatrix.input_names"

        bad_inputs = self.bad_inputs
        self.bad_inputs = [i for i, x in enumerate(self.input_names) if (x in inputs or x in bad_inputs)]

    def enable_inputs(self, inputs: list[str]):
        assert self.input_names is not None, "ResponseMatrix.input_names are not defined"
        for _input in inputs:
            assert _input in self.input_names, f"{_input} not found in ResponseMatrix.input_names"

        bad_inputs = self.bad_inputs
        new_bad_inputs = []
        for bad_input in bad_inputs:
            if self.input_names[bad_input] not in inputs:
                new_bad_inputs.append(bad_input)
        self.bad_inputs = new_bad_inputs

    def disable_all_inputs(self):
        self.disable_inputs(self.input_names)

    def disable_all_inputs_but(self, inputs: list[str]):
        self.disable_all_inputs()
        self.enable_inputs(inputs)

    def enable_all_inputs(self):
        self.bad_inputs = []

    def disable_outputs(self, outputs: list[str]):
        assert self.output_names is not None, "ResponseMatrix.output_names are not defined"
        for _output in outputs:
            assert _output in self.output_names, f"{_output} not found in ResponseMatrix.output_names"

        bad_outputs = self.bad_outputs
        self.bad_outputs = [i for i, x in enumerate(self.output_names) if (x in outputs or x in bad_outputs)]

    def enable_outputs(self, outputs: list[str]):
        assert self.output_names is not None, "ResponseMatrix.output_names are not defined"
        for _output in outputs:
            assert _output in self.output_names, f"{_output} not found in ResponseMatrix.output_names"

        bad_outputs = self.bad_outputs
        new_bad_outputs = []
        for bad_output in bad_outputs:
            if self.output_names[bad_output] not in outputs:
                new_bad_outputs.append(bad_output)
        self.bad_outputs = new_bad_outputs

    def disable_all_outputs(self):
        self.disable_outputs(self.output_names)

    def disable_all_outputs_but(self, outputs: list[str]):
        self.disable_all_inputs()
        self.enable_inputs(outputs)

    def enable_all_outputs(self):
        self.bad_outputs = []

    def build_pseudoinverse(self, method='svd_cutoff', parameter: float = 0., zerosum: bool = False,
                            rf: bool = False, plane: Optional[PLANE_TYPE] = None):
        logger.info(f'(Re-)Building pseudoinverse RM with {method=} and {parameter=} with {zerosum=}.')
        assert plane is None or plane in ['H', 'V'], f'Unknown plane: {plane}.'
        if plane is None:
            matrix = self.matrix[self._output_mask, :][:, self._input_mask]
        elif plane in ['H', 'V']:
            output_plane_mask = self.get_output_plane_mask(plane)
            input_plane_mask = self.get_input_plane_mask(plane)
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            tot_input_mask = np.logical_and(self._input_mask, input_plane_mask)
            matrix = self.matrix[tot_output_mask, :][:, tot_input_mask]

        if zerosum or rf:
            rows, cols = matrix.shape
            if zerosum:
                rows += 1
            if rf:
                cols += 1
            matrix_to_invert = np.zeros((rows, cols), dtype=float)
            matrix_to_invert[:matrix.shape[0], :matrix.shape[1]] = matrix
            if zerosum:
                matrix_to_invert[-1, :matrix.shape[1]]
            if rf:
                rf_response = self.rf_response
                if plane is not None:
                    rf_response = rf_response[tot_output_mask] # tot_output_mask will have been defined earlier always.

                matrix_to_invert[:matrix.shape[0], -1] = self.rf_weight*rf_response
        else:
            matrix_to_invert = matrix

        U, s_mat, Vh = np.linalg.svd(matrix_to_invert, full_matrices=False)
        if method == 'svd_cutoff':
            cutoff = parameter
            s0 = s_mat[0]
            keep = np.sum(s_mat > cutoff * s0)
            d_mat = 1. / s_mat[:keep]
        elif method == 'svd_values':
            number_of_values_to_keep = parameter
            s_mat[int(number_of_values_to_keep):] = 0
            keep = number_of_values_to_keep
            d_mat = 1. / s_mat[:keep]
        elif method == 'tikhonov':
            alpha = parameter
            d_mat = s_mat / (np.square(s_mat) + alpha**2)
            keep = len(d_mat)

        #matrix_inv = np.dot(np.dot(np.transpose(Vh), np.diag(d_mat)), np.transpose(U))
        matrix_inv = np.dot(np.dot(np.transpose(Vh[:keep,:]), np.diag(d_mat)), np.transpose(U[:, :keep]))

        return InverseResponseMatrix(matrix=matrix_inv, method=method, parameter=parameter, zerosum=zerosum)

    def solve(self, output: np.array, method: str = 'svd_cutoff', parameter: float = 0.,
              zerosum: bool = False, rf: bool = False, plane: Optional[Literal['H', 'V']] = None) -> np.ndarray:
        assert len(self.bad_outputs) != self.matrix.shape[0], 'All outputs are disabled!'
        assert len(self.bad_inputs) != self.matrix.shape[1], 'All inputs are disabled!'
        assert plane is None or plane in ['H', 'V'], f'Unknown plane: {plane}.'
        if plane is None:
            expected_shape = (self._n_inputs - len(self._bad_inputs), self._n_outputs - len(self._bad_outputs))
        else:
            output_plane_mask = np.array(self.outputs_plane) == plane
            input_plane_mask = np.array(self.inputs_plane) == plane
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            tot_input_mask = np.logical_and(self._input_mask, input_plane_mask)
            expected_shape = (sum(tot_input_mask), sum(tot_output_mask))

        if zerosum:
            expected_shape = (expected_shape[0], expected_shape[1] + 1)

        if rf:
            expected_shape = (expected_shape[0] + 1, expected_shape[1])


        if method != 'micado':
            if plane is None:
                active_inverse_RM = self._inverse_RM
            elif plane == 'H':
                active_inverse_RM = self._inverse_RM_H
            elif plane == 'V':
                active_inverse_RM = self._inverse_RM_V

            if active_inverse_RM is None:
                active_inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, zerosum=zerosum, plane=plane, rf=rf)
            else:
                if (active_inverse_RM.method != method or active_inverse_RM.parameter != parameter or
                    active_inverse_RM.zerosum != zerosum or active_inverse_RM.shape != expected_shape):
                    active_inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, zerosum=zerosum, plane=plane, rf=rf)

            # cache it
            if plane is None:
               self._inverse_RM = active_inverse_RM
            elif plane == 'H':
               self._inverse_RM_H = active_inverse_RM
            elif plane == 'V':
               self._inverse_RM_V = active_inverse_RM


        output_plane_mask = np.ones_like(self._output_mask, dtype=bool)
        if plane in ['H', 'V']:
            output_plane_mask = np.array(self.outputs_plane) == plane

        bad_output = output.copy()
        bad_output[np.isnan(bad_output)] = 0
        good_output = bad_output[np.logical_and(self._output_mask, output_plane_mask)]


        if method == 'micado':
            bad_input = self.micado(good_output, int(parameter), plane=plane)
            if zerosum:
                logger.warning('Zerosum option is incompatible with the micado method and will be ignored.')
            if rf:
                logger.warning('Rf option is incompatible with the micado method and will be ignored.')
        else:
            if active_inverse_RM.shape != expected_shape:
                raise Exception('Error: shapes of Response matrix, excluding bad inputs and outputs do not match: \n' 
                 + f'inverse RM shape = {active_inverse_RM.shape},\n'
                 + f'expected inputs = {expected_shape[0]},\n'
                 + f'expected outputs = {expected_shape[1]},')

            if zerosum:
                zerosum_good_output = np.zeros(len(good_output) + 1)
                zerosum_good_output[:-1] = good_output
                good_input = active_inverse_RM.dot(zerosum_good_output)
            else:
                good_input = active_inverse_RM.dot(good_output)

            if rf: # split rf from the other inputs
                rf_input = good_input[-1]
                good_input = good_input[:-1]

            final_input_length = self._n_inputs
            if rf:
                final_input_length += 1

            bad_input = np.zeros(final_input_length, dtype=float)
            input_plane_mask = np.ones_like(self._input_mask, dtype=bool)
            if plane in ['H', 'V']:
                input_plane_mask = np.array(self.inputs_plane) == plane
            bad_input[:self._n_inputs][np.logical_and(self._input_mask, input_plane_mask)] = good_input

            if rf:
                bad_input[-1] = rf_input
        return bad_input

    def micado(self, good_output: np.array, n: int, plane: Optional[PLANE_TYPE] = None) -> np.ndarray:
        all_inputs = list(range(self._n_inputs))
        bad_input = np.zeros(self._n_inputs, dtype=float)
        already_used_inputs = []
        if plane is None:
            tot_output_mask = self._output_mask
        else:
            output_plane_mask = self.get_output_plane_mask(plane)
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            input_plane_mask = self.get_input_plane_mask(plane)
            all_inputs = np.array(all_inputs, dtype=int)[input_plane_mask]
        good_matrix = self.matrix[tot_output_mask]

        residual = good_output.copy()

        for _ in range(n):
            best_chi2 = np.inf
            for ii in all_inputs:
                if ii in already_used_inputs or ii in self._bad_inputs:
                    continue
                response = good_matrix[:, ii]
                trim = np.dot(response, residual) / np.dot(response, response)
                chi2 = np.sum(np.square(residual - trim * response))
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_input = ii
                    best_trim = trim
            already_used_inputs.append(best_input)
            bad_input[best_input] = best_trim
            residual -= best_trim * good_matrix[:, best_input]

        return bad_input

    @classmethod
    def from_json(cls, json_filename: str) -> "ResponseMatrix":
        with open(json_filename, 'r') as fp:
            obj = json.load(fp)
            if 'RM' in obj: ## for backwards compatibility, to be removed when RM is completely phased out
                obj['matrix'] = obj['RM']
                del obj['RM']
            return cls.model_validate(obj)

    def to_json(self, json_filename: str) -> None:
        with open(json_filename, 'w') as fp:
            json.dump(self.model_dump(), fp)
