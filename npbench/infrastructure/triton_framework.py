# Copyright 2025 ETH Zurich and the NPBench authors. All rights reserved.
import pkg_resources

from npbench.infrastructure import Benchmark, Framework
from typing import Any, Callable, Dict


class TritonFramework(Framework):
    """ A class for reading and processing framework information. """

    def __init__(self, fname: str):
        """ Reads framework information.
        :param fname: The framework name.
        """

        super().__init__(fname)

    def version(self) -> str:
        """ Return the framework version. """
        return pkg_resources.get_distribution("triton").version

    def imports(self) -> Dict[str, Any]:
        return {"torch": __import__("torch")}

    def copy_func(self) -> Callable:
        import torch
        torch.set_default_device('cuda')
        def inner(arr):
            copy = torch.from_numpy(arr).to('cuda')
            return copy
        return inner

    def exec_str(self, bench: Benchmark, impl: Callable = None):
        """ Generates the execution-string that should be used to call
        the benchmark implementation.
        :param bench: A benchmark.
        :param impl: A benchmark implementation.
        """

        return f"__npb_result = __npb_impl({self.arg_str(bench, impl)}); torch.cuda.synchronize()"
