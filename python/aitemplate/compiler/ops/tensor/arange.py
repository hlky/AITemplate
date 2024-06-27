from typing import List, Union

from aitemplate import backend
from aitemplate.backend import registry
from aitemplate.compiler.base import IntVar, Operator, Tensor


class arange(Operator):
    def __init__(
        self,
        start: Union[int, IntVar],
        stop: Union[int, IntVar],
        step: Union[int, IntVar],
    ):
        super().__init__()
        self._attrs["op"] = "arange"
        self._attrs["has_profiler"] = False
        self._attrs["nop"] = False
        self._attrs["start"] = start if isinstance(start, IntVar) else IntVar([start])
        self._attrs["stop"] = stop if isinstance(stop, IntVar) else IntVar([stop])
        self._attrs["step"] = step if isinstance(step, IntVar) else IntVar([step])

    def __call__(self) -> Tensor:
        self._attrs["inputs"] = []
        self._set_depth()
        output_shape = self._infer_shape()
        output = Tensor(output_shape, src_ops={self})
        self._attrs["outputs"] = [output]
        return output

    def _infer_shape(self) -> List[IntVar]:
        num_elements = (self._attrs["stop"] - self._attrs["start"]) / self._attrs[
            "step"
        ]
        return [num_elements]

    def gen_function(self) -> str:
        target = backend.target.Target.current()
        func_key = f"{target.name()}.{self._attrs['op']}.gen_function"
        func = registry.get(func_key)
        return func(self._attrs)
