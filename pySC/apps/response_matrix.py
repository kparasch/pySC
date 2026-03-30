from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Literal, Any
from ..core.types import NPARRAY
import numpy as np
import logging
import json

## Some timing info with 640 outputs, 576 inputs:
## response_matrix.build_pseudoinverse() -> 30 ms
## response_matrix.solve() -> 2 ms (if pseudo-inverse is cached)
## hash(bytes(response_matrix.input_weights)) -> 0.2 ms
## hash(bytes(response_matrix.output_weights)) -> 0.2 ms
##


PLANE_TYPE = Literal['H', 'V', 'Q', 'SQ']

logger = logging.getLogger(__name__)

class InverseResponseMatrix(BaseModel, extra="forbid"):
    matrix: NPARRAY
    method: Literal['tikhonov', 'svd_values', 'svd_cutoff', 'micado']
    parameter: float
    virtual: bool = True
    rf: bool = False
    rf_weight: float
    virtual_weight: float
    hash_rf_response: int
    hash_input_weights: int
    hash_output_weights: int

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
    input_planes: Optional[list[PLANE_TYPE]] = None
    output_planes: Optional[list[PLANE_TYPE]] = None

    rf_response: Optional[NPARRAY] = None
    input_weights: Optional[NPARRAY] = None
    output_weights: Optional[NPARRAY] = None
    rf_weight: Optional[float] = None
    virtual_weight: float = 1

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
    _rf_weight_is_default: bool = PrivateAttr(default=True)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def RM(self):
        logger.warning('ResponseMatrix.RM is deprecated! Please use ResponseMatrix.matrix instead.')
        return self.matrix

    @model_validator(mode='before')
    def check_deprecations(cls, data):
        if 'inputs_plane' in data.keys():
            logger.warning('DEPRECATION: `inputs_plane` in the ResponseMatrix has been renamed to `input_planes`.')
            logger.warning('DEPRECATION: You should do the same.')
            if 'input_planes' in data.keys():
                raise Exception('Both `inputs_plane` and `input_planes` are in the ResponseMatrix. Please only use the later one.')
            else:
                logger.warning('DEPRECATION: renaming automatically `inputs_plane` to `input_planes`.')
                data['input_planes'] = data['inputs_plane']
                del data['inputs_plane']

        if 'outputs_plane' in data.keys():
            logger.warning('DEPRECATION: `outputs_plane` in the ResponseMatrix has been renamed to `output_planes`.')
            logger.warning('DEPRECATION: You should do the same.')
            if 'output_planes' in data.keys():
                raise Exception('Both `outputs_plane` and `output_planes` are in the ResponseMatrix. Please only use the later one.')
            else:
                logger.warning('DEPRECATION: renaming automatically `outputs_plane` to `output_planes`.')
                data['output_planes'] = data['outputs_plane']
                del data['outputs_plane']
        return data

    @model_validator(mode='after')
    def initialize_and_check(self):
        self._n_outputs, self._n_inputs = self.matrix.shape
        try:
            self._singular_values = np.linalg.svd(self.matrix, compute_uv=False)
        except np.linalg.LinAlgError:
            logger.warning('SVD of the response matrix failed, correction will be impossible.')
            self._singular_values = None
        self.make_masks()

        if self.input_planes is None:
            Nh = self._n_inputs // 2
            if self._n_inputs % 2 != 0:
                logger.warning('Plane of inputs is undefined and number of inputs in response matrix is not even. '
                               'Misinterpretation of the input plane is guaranteed!')
            self.input_planes = ['H'] * Nh + ['V'] * (self._n_inputs - Nh)

        if self.output_planes is None:
            Nh = self._n_outputs // 2
            if self._n_outputs % 2 != 0:
                logger.warning('Plane of outputs is undefined and number of outputs in response matrix is not even. '
                               'Misinterpretation of the output plane is guaranteed!')
            self.output_planes = ['H'] * Nh + ['V'] * (self._n_outputs - Nh)

        user_provided_rf_weight = self.rf_weight is not None
        self._rf_weight_is_default = not user_provided_rf_weight
        if self.rf_response is None:
            self.rf_response = np.zeros(self._n_outputs)
        else:
            if len(self.rf_response) != self._n_outputs:
                logger.warning(f'RF response does not have the correct length: {len(self.rf_response)} (should have been {self._n_outputs}).')
                logger.warning('RF response will be removed.')
                self.rf_response = np.zeros(self._n_outputs)
            else: # self.rf_response is not None and is valid
                if not user_provided_rf_weight:
                    default_rf_weight = self.default_rf_weight()
                    logger.info(f'Setting the rf_weight by default to {default_rf_weight}.')
                    self.rf_weight = default_rf_weight

        if self.rf_weight is None: #if it is still None, rf_response is not valid and set it to 0.
            self.rf_weight = 0

        if self.input_weights is None:
            self.input_weights = np.ones(self._n_inputs, dtype=float)

        if self.output_weights is None:
            self.output_weights = np.ones(self._n_outputs, dtype=float)

        return self

    @property
    def hash_rf_response(self) -> int:
        return hash(bytes(self.rf_response))

    @property
    def hash_input_weights(self) -> int:
        return hash(bytes(self.input_weights))

    @property
    def hash_output_weights(self) -> int:
        return hash(bytes(self.output_weights))

    def set_weight(self, name: str, weight: float, plane: Optional[PLANE_TYPE] = None):
        applied = False
        for ii, input_name in enumerate(self.input_names):
            if input_name == name:
                if plane is None or self.input_planes[ii] == plane:
                    self.input_weights[ii] = weight
                    applied = True

        for ii, output_name in enumerate(self.output_names):
            if output_name == name:
                if plane is None or self.output_planes[ii] == plane:
                    self.output_weights[ii] = weight
                    applied = True

        if not applied:
            logger.warning('{name} was not found to apply weight.')

        return

    def default_rf_weight(self) -> float:
        if self.rf_response is None:
            raise Exception('rf_response was not found.')
        matrix_h = self.matrix_h
        rms_per_input = np.std(matrix_h, axis=0)
        mean_rms_inputs = np.mean(rms_per_input)
        rms_rf = np.std(self.rf_response)
        default_rf_weight = mean_rms_inputs / rms_rf
        return default_rf_weight

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
        if self._rf_weight_is_default:
            self.rf_weight = self.default_rf_weight()
            logger.info(f'Setting the rf_weight by default to {self.rf_weight}, after rf_response was changed.')
        return

    def get_matrix_in_plane(self, plane: Optional[PLANE_TYPE] = None) -> np.array:
        if plane is None:
            return self.matrix
        else:
            output_plane_mask = self.get_output_plane_mask(plane)
            input_plane_mask = self.get_input_plane_mask(plane)
            return self.matrix[output_plane_mask, :][:, input_plane_mask]

    def get_input_plane_mask(self, plane: Literal[PLANE_TYPE]) -> np.array:
        return np.array(self.input_planes) == plane

    def get_output_plane_mask(self, plane: Literal[PLANE_TYPE]) -> np.array:
        return np.array(self.output_planes) == plane

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
        self.disable_all_outputs()
        self.enable_outputs(outputs)

    def enable_all_outputs(self):
        self.bad_outputs = []

    def build_pseudoinverse(self, method='svd_cutoff', parameter: float = 0., virtual: bool = False,
                            rf: bool = False, plane: Optional[PLANE_TYPE] = None):
        logger.info(f'(Re-)Building pseudoinverse RM with {method=}, {parameter=}, {plane=}, {virtual=}, {rf=}.')
        assert plane is None or plane in ['H', 'V'], f'Unknown plane: {plane}.'
        if plane is None:
            matrix = self.matrix[self._output_mask, :][:, self._input_mask]
            tot_output_mask = self._output_mask
            tot_input_mask = self._input_mask
        elif plane in ['H', 'V']:
            output_plane_mask = self.get_output_plane_mask(plane)
            input_plane_mask = self.get_input_plane_mask(plane)
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            tot_input_mask = np.logical_and(self._input_mask, input_plane_mask)
            matrix = self.matrix[tot_output_mask, :][:, tot_input_mask]

        # at this point, matrix has bad outputs/inputs (bpms/correctors) removed.
        # also has only plane-specific outputs/inputs if requested.

        # add extra column/row for the rf response or to enforce that the sum of all outputs is zero.
        if virtual or rf:
            rows, cols = matrix.shape
            if virtual:
                rows += 1
            if rf:
                cols += 1
            matrix_to_invert = np.zeros((rows, cols), dtype=float)
            matrix_to_invert[:matrix.shape[0], :matrix.shape[1]] = matrix
            if virtual:
                matrix_to_invert[-1, :matrix.shape[1]] = self.virtual_weight
            if rf:
                rf_response = self.rf_response
                if plane is not None:
                    rf_response = rf_response[tot_output_mask] # tot_output_mask will have been defined earlier always.

                matrix_to_invert[:matrix.shape[0], -1] = self.rf_weight * rf_response
        else:
            matrix_to_invert = matrix

        # matrix_to_invert is extended by one row and/or column if rf and/or virtual was enabled.

        # handle weights
        rows, cols = matrix.shape
        matrix_to_invert[:, :cols] = np.multiply(matrix_to_invert[:, :cols], self.input_weights[tot_input_mask])
        matrix_to_invert[:rows, :] = np.multiply(matrix_to_invert[:rows, :], self.output_weights[tot_output_mask][:, np.newaxis])

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

        return InverseResponseMatrix(matrix=matrix_inv, method=method, parameter=parameter, virtual=virtual,
                                     virtual_weight=self.virtual_weight, rf=rf, rf_weight=self.rf_weight,
                                     hash_rf_response=self.hash_rf_response,
                                     hash_input_weights=self.hash_input_weights,
                                     hash_output_weights=self.hash_output_weights)

    def solve(self, output: np.array, method: str = 'svd_cutoff', parameter: float = 0., virtual: bool = False,
              zerosum: Optional[bool] = None, rf: bool = False, plane: Optional[Literal['H', 'V']] = None,
              virtual_target: float = 0, solver: Optional[Any] = None) -> np.ndarray:
        if zerosum is not None:
            logger.warning('`zerosum` argument in ResponseMatrix.solve is deprecated. Please use `virtual` instead.')
            virtual = zerosum

        assert len(self.bad_outputs) != self.matrix.shape[0], 'All outputs are disabled!'
        assert len(self.bad_inputs) != self.matrix.shape[1], 'All inputs are disabled!'
        assert plane is None or plane in ['H', 'V'], f'Unknown plane: {plane}.'
        if plane is None:
            expected_shape = (self._n_inputs - len(self._bad_inputs), self._n_outputs - len(self._bad_outputs))
        else:
            output_plane_mask = np.array(self.output_planes) == plane
            input_plane_mask = np.array(self.input_planes) == plane
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            tot_input_mask = np.logical_and(self._input_mask, input_plane_mask)
            expected_shape = (sum(tot_input_mask), sum(tot_output_mask))

        if virtual:
            expected_shape = (expected_shape[0], expected_shape[1] + 1)

        if rf:
            expected_shape = (expected_shape[0] + 1, expected_shape[1])


        if solver is None and method != 'micado':
            if plane is None:
                active_inverse_RM = self._inverse_RM
            elif plane == 'H':
                active_inverse_RM = self._inverse_RM_H
            elif plane == 'V':
                active_inverse_RM = self._inverse_RM_V

            if active_inverse_RM is None:
                active_inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, virtual=virtual, plane=plane, rf=rf)
            else:
                if not (active_inverse_RM.method == method
                        and active_inverse_RM.parameter == parameter
                        and active_inverse_RM.virtual == virtual
                        and active_inverse_RM.virtual_weight == self.virtual_weight
                        and active_inverse_RM.rf == rf
                        and active_inverse_RM.shape == expected_shape
                        and active_inverse_RM.hash_rf_response == self.hash_rf_response
                        and active_inverse_RM.rf_weight == self.rf_weight
                        and active_inverse_RM.hash_input_weights == self.hash_input_weights
                        and active_inverse_RM.hash_output_weights == self.hash_output_weights
                       ):
                    active_inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, virtual=virtual, plane=plane, rf=rf)

            # cache it
            if plane is None:
               self._inverse_RM = active_inverse_RM
            elif plane == 'H':
               self._inverse_RM_H = active_inverse_RM
            elif plane == 'V':
               self._inverse_RM_V = active_inverse_RM


        output_plane_mask = np.ones_like(self._output_mask, dtype=bool)
        if plane in ['H', 'V']:
            output_plane_mask = np.array(self.output_planes) == plane

        bad_output = output.copy() * self.output_weights
        bad_output[np.isnan(bad_output)] = 0
        good_output = bad_output[np.logical_and(self._output_mask, output_plane_mask)]


        if solver is not None:
            matrix, _ = self._solver_matrix(virtual=virtual, rf=rf, plane=plane)
            if virtual:
                external_output = np.zeros(len(good_output) + 1)
                external_output[:-1] = good_output
                external_output[-1] = virtual_target * self.virtual_weight
            else:
                external_output = good_output
            good_input = self._solver_fit(solver, matrix, external_output)
            if rf:
                rf_input = good_input[-1]
                good_input = good_input[:-1]
        elif method == 'micado':
            bad_input = self.micado(good_output, int(parameter), plane=plane)
            if virtual:
                logger.warning('virtual option is incompatible with the micado method and will be ignored.')
            if rf:
                logger.warning('Rf option is incompatible with the micado method and will be ignored.')
        else:
            if active_inverse_RM.shape != expected_shape:
                raise Exception('Error: shapes of Response matrix, excluding bad inputs and outputs do not match: \n' 
                 + f'inverse RM shape = {active_inverse_RM.shape},\n'
                 + f'expected inputs = {expected_shape[0]},\n'
                 + f'expected outputs = {expected_shape[1]},')

            if virtual:
                virtual_good_output = np.zeros(len(good_output) + 1)
                virtual_good_output[:-1] = good_output
                virtual_good_output[-1] = virtual_target * self.virtual_weight
                good_input = active_inverse_RM.dot(virtual_good_output)
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
                input_plane_mask = np.array(self.input_planes) == plane
            bad_input[:self._n_inputs][np.logical_and(self._input_mask, input_plane_mask)] = good_input

            bad_input[:self._n_inputs] = np.multiply(bad_input[:self._n_inputs], self.input_weights)
            if rf:
                bad_input[-1] = rf_input * self.rf_weight
        if solver is not None:
            final_input_length = self._n_inputs
            if rf:
                final_input_length += 1

            bad_input = np.zeros(final_input_length, dtype=float)
            if plane in ['H', 'V']:
                input_plane_mask = np.array(self.input_planes) == plane
            else:
                input_plane_mask = np.ones_like(self._input_mask, dtype=bool)
            bad_input[:self._n_inputs][np.logical_and(self._input_mask, input_plane_mask)] = good_input
            bad_input[:self._n_inputs] = np.multiply(bad_input[:self._n_inputs], self.input_weights)
            if rf:
                bad_input[-1] = rf_input * self.rf_weight
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
            best_input = None
            best_trim = None
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
            if best_input is None:
                break
            already_used_inputs.append(best_input)
            bad_input[best_input] = best_trim
            residual -= best_trim * good_matrix[:, best_input]

        return bad_input


    def _solver_matrix(self, virtual: bool = False, rf: bool = False, plane: Optional[PLANE_TYPE] = None) -> tuple[np.ndarray, np.ndarray]:
        assert plane is None or plane in ['H', 'V'], f'Unknown plane: {plane}.'
        if plane is None:
            matrix = self.matrix[self._output_mask, :][:, self._input_mask]
            tot_output_mask = self._output_mask
            tot_input_mask = self._input_mask
        else:
            output_plane_mask = self.get_output_plane_mask(plane)
            input_plane_mask = self.get_input_plane_mask(plane)
            tot_output_mask = np.logical_and(self._output_mask, output_plane_mask)
            tot_input_mask = np.logical_and(self._input_mask, input_plane_mask)
            matrix = self.matrix[tot_output_mask, :][:, tot_input_mask]

        if virtual or rf:
            rows, cols = matrix.shape
            if virtual:
                rows += 1
            if rf:
                cols += 1
            matrix_to_solve = np.zeros((rows, cols), dtype=float)
            matrix_to_solve[:matrix.shape[0], :matrix.shape[1]] = matrix
            if virtual:
                matrix_to_solve[-1, :matrix.shape[1]] = self.virtual_weight
            if rf:
                rf_response = self.rf_response
                if plane is not None:
                    rf_response = rf_response[tot_output_mask]
                matrix_to_solve[:matrix.shape[0], -1] = self.rf_weight * rf_response
        else:
            matrix_to_solve = matrix.copy()

        rows, cols = matrix.shape
        matrix_to_solve[:, :cols] = np.multiply(matrix_to_solve[:, :cols], self.input_weights[tot_input_mask])
        matrix_to_solve[:rows, :] = np.multiply(matrix_to_solve[:rows, :], self.output_weights[tot_output_mask][:, np.newaxis])
        return matrix_to_solve, tot_input_mask

    @staticmethod
    def _solver_fit(solver: Any, matrix: np.ndarray, output: np.ndarray) -> np.ndarray:
        if not hasattr(solver, 'fit'):
            raise TypeError('External solver must define a fit(X, y) method.')
        solver.fit(matrix, output)
        if not hasattr(solver, 'coef_'):
            raise TypeError('External solver must expose fitted coefficients via `coef_`.')
        good_input = np.asarray(solver.coef_, dtype=float)
        if good_input.ndim != 1:
            good_input = good_input.reshape(-1)
        if good_input.shape[0] != matrix.shape[1]:
            raise ValueError(
                f'External solver returned {good_input.shape[0]} coefficients, expected {matrix.shape[1]}.'
            )
        return good_input


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
