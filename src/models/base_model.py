from torch import nn, Tensor
from abc import abstractmethod
from typing import Any


class BaseModel(nn.Module):
    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_function(self, batch: Tensor, *inputs: Any, **kwargs) -> Tensor:
        raise NotImplementedError
