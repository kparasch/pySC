
from pydantic import BeforeValidator, PlainSerializer
from typing import Annotated
import numpy as np

NPARRAY = Annotated[np.ndarray,
                    BeforeValidator(lambda x: np.array(x)),
                    PlainSerializer(lambda x: x.tolist(), return_type=list)
                   ]