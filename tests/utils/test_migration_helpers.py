import warnings
from typing import Tuple

import pytest

from kartothek.utils.migration_helpers import (
    deprecate_parameters,
    deprecate_parameters_if_set,
)

MESSAGE = "Parameter {parameter} is deprecated!"


@deprecate_parameters(MESSAGE, "param1")
def func_non_optional_params_one_param(param1: int):
    return param1


@deprecate_parameters(MESSAGE, "param2", "param3")
def func_non_optional_params_multiple_params(
    param1: int, param2: int, param3: int
) -> Tuple[int, int, int]:
    return param1, param2, param3


@deprecate_parameters_if_set(MESSAGE, "param1", "param2")
def func_optional_params_mixed_optional(
    param1: int, param2: int = None
) -> Tuple[int, int]:
    return param1, param2


@deprecate_parameters_if_set(MESSAGE, "param2", "param3")
def func_optional_params_multiple_params(
    param1: int, param2: int = None, param3: int = None
) -> Tuple[int, int, int]:
    return param1, param2, param3


def test_deprecate_parameter_multi_table():
    # check: init raises without passed parameters
    with pytest.raises(ValueError):

        @deprecate_parameters(MESSAGE)
        def func_test_raise_unparametrized_decorator(param1: int) -> int:
            return param1

    # check: init raises duplicate param
    with pytest.raises(ValueError):

        @deprecate_parameters(MESSAGE, "param1", "param1")
        def func_test_raise_duplicate_parameter_decorator(
            param1: int, param2: int
        ) -> Tuple[int, int]:
            return param1, param2

    # check: init raises declared param does not match function signature
    with pytest.raises(ValueError):

        @deprecate_parameters(MESSAGE, "param100000000")
        def func_test_raise_invalid_parameter_decorator(param1: int) -> int:
            return param1

    # check args
    with pytest.warns(DeprecationWarning, match="Parameter param1 is deprecated!"):
        result = func_non_optional_params_one_param(0)
        assert result == 0
    # check args and kwargs
    with pytest.warns(DeprecationWarning, match="Parameter param1 is deprecated!"):
        result = func_non_optional_params_one_param(param1=1)
        assert result == 1

    # check args
    with pytest.warns(DeprecationWarning, match="is deprecated!"):
        result = func_non_optional_params_multiple_params(0, 1, 2)
        assert result == (0, 1, 2)
    # check args and kwargs
    with pytest.warns(DeprecationWarning, match="is deprecated!"):
        result = func_non_optional_params_multiple_params(0, param2=1, param3=2)
        assert result == (0, 1, 2)


def test_deprecate_optional_parameter_if_set_multi_table():
    # check: init raises without passed parameters
    with pytest.raises(ValueError):

        @deprecate_parameters_if_set(MESSAGE)
        def func_test_raise_unparametrized_decorator(param1: int) -> int:
            return param1

    # check: init raises duplicate param
    with pytest.raises(ValueError):

        @deprecate_parameters_if_set(MESSAGE, "param1", "param1")
        def func_test_raise_duplicate_parameter_decorator(
            param1: int, param2: int
        ) -> Tuple[int, int]:
            return param1, param2

    # check: init raises declared param does not match function signature
    with pytest.raises(ValueError):

        @deprecate_parameters_if_set(MESSAGE, "param100000000")
        def func_test_raise_invalid_parameter_decorator(param1: int) -> int:
            return param1

    # check: do NOT warn when deprecated params not specified
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = func_optional_params_multiple_params(0)
        assert result == (0, None, None)

    # check: raise if used without kwargs
    with pytest.warns(DeprecationWarning,):
        result = func_optional_params_multiple_params(0, 1, 2)
        assert result == (0, 1, 2)

    # check: raise if used with kwargs
    with pytest.warns(DeprecationWarning,):
        result = func_optional_params_multiple_params(0, param2=1, param3=2)
        assert result == (0, 1, 2)

    # check: raise mixed if used without kwargs
    with pytest.warns(DeprecationWarning,):
        result = func_optional_params_mixed_optional(0, 1)
        assert result == (0, 1)

    # check: raise mixed if used without kwargs
    with pytest.warns(DeprecationWarning,):
        result = func_optional_params_mixed_optional(param1=0, param2=1)
        assert result == (0, 1)
