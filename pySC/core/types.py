from pydantic import BeforeValidator, PlainSerializer, BaseModel
from typing import Annotated, Union, Optional
import numpy as np
from pathlib import Path

NPARRAY = Annotated[np.ndarray,
                    BeforeValidator(lambda x: np.array(x)),
                    PlainSerializer(lambda x: x.tolist(), return_type=list)
                   ]

class BaseModelWithSave(BaseModel):
    def save_as(self, filename: Union[Path, str], indent: Optional[int] = None) -> None:
        if type(filename) is not Path:
            filename = Path(filename)

        data = self.model_dump()
        suffix = filename.suffix
        with open(filename, 'w') as fp:
            if suffix in ['', '.json']:
                import json
                json.dump(data, fp, indent=indent)
            elif suffix == '.yaml':
                import yaml
                yaml.safe_dump(data, fp, indent=indent)
            else:
                raise Exception(f'Unknown file extension: {suffix}.')
        return