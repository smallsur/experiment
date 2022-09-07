# from typing import Union, Sequence, Tuple
from typing import Union
from typing import Sequence
from typing import Tuple

import torch

TensorOrSequence = Union[Sequence[torch.Tensor], torch.Tensor]
TensorOrNone = Union[torch.Tensor, None]


