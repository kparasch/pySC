from pydantic import BaseModel, PrivateAttr, model_validator, ConfigDict
from typing import Optional, Literal
import numpy as np
import logging
import json

from ..core.numpy_type import NPARRAY

logger = logging.getLogger(__name__)

class InverseResponseMatrix(BaseModel, extra="forbid"):
    matrix: NPARRAY
    method: Literal['tikhonov', 'svd_values', 'svd_cutoff', 'micado']
    parameter: float
    zerosum: bool = True

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
    inputs_plane: Optional[list[Literal['H','V']]] = None

    _n_outputs: int = PrivateAttr(default=0)
    _n_inputs: int = PrivateAttr(default=0)
    _singular_values: Optional[NPARRAY] = PrivateAttr(default=None)
    _bad_outputs: list[int] = PrivateAttr(default=[])
    _bad_inputs: list[int] = PrivateAttr(default=[])

    _output_mask: NPARRAY = np.array([])
    _inverse_RM: Optional[InverseResponseMatrix] = PrivateAttr(default=None)

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
            if Nh % 2 != 0:
                logger.warning('Plane of inputs is undefined and number of inputs in response matrix is not even.\
                                Misinterpretation of the input plane is guaranteed!')
            self.inputs_plane = ['H'] * Nh + ['V'] * (self._n_inputs - Nh)
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
    def bad_outputs(self) -> list[int]:
        return self._bad_outputs

    @bad_outputs.setter
    def bad_outputs(self, bad_list: list[int]) -> None:
        self._bad_outputs = bad_list.copy()
        self.make_masks()

    def make_masks(self):
        self._inverse_RM = None # discard inverse RM, by changing bad inputs/outputs it becomes invalid
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

    def build_pseudoinverse(self, method='svd_cutoff', parameter: float = 0., zerosum: bool = False):
        logger.info(f'(Re-)Building pseudoinverse RM with {method=} and {parameter=} with {zerosum=}.')
        matrix = self.matrix[self._output_mask, :][:, self._input_mask]

        if zerosum:
            # select only horizontal plane to zero-sum the inputs. Should we do for any plane?
            zerosummed_matrix = np.zeros([matrix.shape[0] + 1, matrix.shape[1]])
            zerosummed_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
            horizontal_mask = np.array(self.inputs_plane)[self._input_mask] == 'H'
            zerosummed_matrix[-1][horizontal_mask] = 1
            U, s_mat, Vh = np.linalg.svd(zerosummed_matrix, full_matrices=False)
        else:
            U, s_mat, Vh = np.linalg.svd(matrix, full_matrices=False)

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

        return InverseResponseMatrix(matrix=matrix_inv, method=method, parameter=parameter)

    def solve(self, output: np.array, method: str = 'svd_cutoff', parameter: float = 0., zerosum: bool = False) -> np.ndarray:
        assert len(self.bad_outputs) != self.matrix.shape[0], 'All outputs are disabled!'
        assert len(self.bad_inputs) != self.matrix.shape[1], 'All inputs are disabled!'
        expected_shape = (self._n_inputs - len(self._bad_inputs), self._n_outputs - len(self._bad_outputs))
        if zerosum:
            expected_shape = (expected_shape[0], expected_shape[1] + 1)

        if method != 'micado':
            if self._inverse_RM is None:
                self._inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, zerosum=zerosum)
            else:
                if self._inverse_RM.method != method or self._inverse_RM.parameter != parameter or self._inverse_RM.zerosum != zerosum or self._inverse_RM.shape != expected_shape:
                    self._inverse_RM = self.build_pseudoinverse(method=method, parameter=parameter, zerosum=zerosum)

        bad_output = output.copy()
        bad_output[np.isnan(bad_output)] = 0
        good_output = bad_output[self._output_mask]


        if method == 'micado':
            bad_input = self.micado(good_output, int(parameter))
            if zerosum:
                logger.warning('Zerosum option is incompatible with the micado method and will be ignored.')
        else:
            if self._inverse_RM.shape != expected_shape:
                raise Exception('Error: shapes of Response matrix, excluding bad inputs and outputs do not match: \n' 
                 + f'inverse RM shape = {self._inverse_RM.shape},\n'
                 + f'expected inputs: {self._n_inputs} - {len(self._bad_inputs)},\n'
                 + f'expected outputs: {self._n_outputs} - {len(self._bad_outputs)}\n'
                 + f'received outputs: {len(output)}, should be equal to {self._n_outputs}!')

            if zerosum:
                zerosum_good_output = np.zeros(len(good_output) + 1)
                zerosum_good_output[:-1] = good_output
                good_input = self._inverse_RM.dot(zerosum_good_output)
            else:
                good_input = self._inverse_RM.dot(good_output)

            bad_input = np.zeros(self._n_inputs, dtype=float)
            bad_input[self._input_mask] = good_input
        return bad_input

    def micado(self, good_output: np.array, n: int) -> np.ndarray:
        all_inputs = list(range(self._n_inputs))
        bad_input = np.zeros(self._n_inputs, dtype=float)
        already_used_inputs = []

        good_matrix = self.matrix[self._output_mask]
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