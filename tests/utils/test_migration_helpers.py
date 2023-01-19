import inspect
import warnings
from typing import Callable, Tuple

import pytest

from kartothek.utils.migration_helpers import (
    DEPRECATION_WARNING_PARAMETER_NON_OPTIONAL_GENERIC_VERSION,
    DEPRECATION_WARNING_REMOVE_FUNCTION_GENERIC_VERSION,
    DEPRECATION_WARNING_REMOVE_PARAMETER,
    _WarningActuator,
    deprecate_parameters,
    deprecate_parameters_if_set,
    get_deprecation_warning_parameter_non_optional,
    get_deprecation_warning_remove_dict_multi_table,
    get_deprecation_warning_remove_parameter_multi_table,
    get_generic_function_deprecation_waring,
    get_parameter_default_value_deprecation_warning,
    get_parameter_generic_replacement_deprecation_warning,
    get_parameter_replaced_by_deprecation_warning,
    get_parameter_type_change_deprecation_warning,
    get_specific_function_deprecation_warning,
    get_specific_function_deprecation_warning_multi_table,
)

MESSAGE = "Parameter {parameter} is deprecated!"


@pytest.fixture(
    scope="class",
    params=[
        DEPRECATION_WARNING_REMOVE_PARAMETER,
        DEPRECATION_WARNING_PARAMETER_NON_OPTIONAL_GENERIC_VERSION,
    ],
)
def default_parameter_deprecation_texts_generic(request) -> str:
    return request.param


@pytest.fixture(
    scope="class",
    params=[DEPRECATION_WARNING_REMOVE_FUNCTION_GENERIC_VERSION],
)
def default_function_deprecation_texts_generic(request) -> str:
    return request.param


@pytest.fixture(
    scope="class",
    params=[get_generic_function_deprecation_waring(function_name="test_func")],
)
def default_deprecation_texts_generic(request) -> str:
    return request.param


@pytest.fixture(
    scope="class",
    params=[
        get_deprecation_warning_remove_parameter_multi_table,
        get_deprecation_warning_remove_dict_multi_table,
        get_deprecation_warning_parameter_non_optional,
        get_parameter_replaced_by_deprecation_warning,
        get_parameter_default_value_deprecation_warning,
        get_parameter_type_change_deprecation_warning,
        get_parameter_generic_replacement_deprecation_warning,
    ],
)
def default_param_deprecation_texts_specific(request) -> Callable:
    return request.param


@pytest.fixture(
    scope="class",
    params=[
        get_specific_function_deprecation_warning,
        get_specific_function_deprecation_warning_multi_table,
    ],
)
def default_function_deprecation_texts_specific(request) -> Callable:
    return request.param


@deprecate_parameters_if_set("bla3 {parameter}", "param3")
@deprecate_parameters("bla4 {parameter}", "param4")
def func_non_optional_params_multiple_params_stacked_nested_counterpart(
    param3: int, param4: int
) -> Tuple[int, int]:
    return param3, param4


@deprecate_parameters("bla6 {parameter}", "param6")
@deprecate_parameters_if_set("bla5 {parameter}", "param5")
def func_non_optional_params_multiple_params_stacked_nested_counterpart_inverse(
    param5: int, param6: int
) -> Tuple[int, int]:
    return param5, param6


@pytest.fixture(
    scope="class",
    params=[
        func_non_optional_params_multiple_params_stacked_nested_counterpart,
        func_non_optional_params_multiple_params_stacked_nested_counterpart_inverse,
    ],
)
def func_non_optional_params_multiple_params_stacked_nested_counterparts(
    request,
) -> Callable:
    return request.param


@deprecate_parameters(MESSAGE, "param1")
def func_non_optional_params_one_param(param1: int):
    return param1


@deprecate_parameters(MESSAGE, "param2", "param3")
def func_non_optional_params_multiple_params(
    param1: int, param2: int, param3: int
) -> Tuple[int, int, int]:
    return param1, param2, param3


@deprecate_parameters_if_set("bla1 {parameter}", "param1")
@deprecate_parameters("bla2 {parameter}", "param2")
@deprecate_parameters_if_set("bla3 {parameter}", "param3")
def func_non_optional_params_multiple_params_stacked(
    param1: int, param2: int, param3: int
) -> Tuple[int, int, int]:
    return param1, param2, param3


@deprecate_parameters_if_set("bla1 {parameter}", "param1")
@deprecate_parameters("bla2 {parameter}", "param2")
def func_non_optional_params_multiple_params_stacked_nested(
    func, param1: int, param2: int
) -> Tuple[int, int]:
    return func(param1, param2)


@deprecate_parameters("bla2 {parameter}", "param2")
@deprecate_parameters_if_set("bla1 {parameter}", "param1")
def func_non_optional_params_multiple_params_stacked_nested_inverse(
    func, param1: int, param2: int
) -> Tuple[int, int]:
    return func(param1, param2)


@deprecate_parameters("bla1 {parameter}", "param1")
@deprecate_parameters_if_set("bla2 {parameter}", "param2")
@deprecate_parameters("bla3 {parameter}", "param3")
def func_non_optional_params_multiple_params_stacked_inverse(
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

    # check: init raises no params exist
    with pytest.raises(ValueError):

        @deprecate_parameters(MESSAGE, "param1")
        def func_test_raise_missing_parameter_decorator() -> None:
            pass

    # check args
    with pytest.warns(
        DeprecationWarning, match="Parameter param1 is deprecated!"
    ) as warn_record:
        result = func_non_optional_params_one_param(0)
        assert len(warn_record) == 1
        assert result == 0

    # check args and kwargs
    with pytest.warns(
        DeprecationWarning, match="Parameter param1 is deprecated!"
    ) as warn_record:
        result = func_non_optional_params_one_param(param1=1)
        assert len(warn_record) == 1
        assert result == 1

    # check args
    with pytest.warns(DeprecationWarning, match="is deprecated!") as warn_record:
        result = func_non_optional_params_multiple_params(0, 1, 2)
        assert len(warn_record) == 2
        assert result == (0, 1, 2)

    # check args and kwargs
    with pytest.warns(DeprecationWarning, match="is deprecated!") as warn_record:
        result = func_non_optional_params_multiple_params(0, param2=1, param3=2)
        assert len(warn_record) == 2
        assert result == (0, 1, 2)


def test_deprecate_parameter_stacked():
    # check: raise if used without kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_non_optional_params_multiple_params_stacked(0, 1, 2)

    assert len(warn_record) == 3
    messages = ["bla1 param1", "bla2 param2", "bla3 param3"]
    for i, message in enumerate(messages):
        assert message in warn_record[i].message.args[0]
    assert result == (0, 1, 2)

    with pytest.raises(ValueError):
        # duplicate entry across decorators
        @deprecate_parameters_if_set("bla1 {parameter}", "param1")
        @deprecate_parameters("bla2 {parameter}", "param1")
        @deprecate_parameters_if_set("bla3 {parameter}", "param3")
        def func_non_optional_params_multiple_params_stacked_duplicate(
            param1: int, param2: int, param3: int
        ) -> Tuple[int, int, int]:
            return param1, param2, param3


def test_deprecate_parameter_stacked_inverse():
    # check: raise if used without kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_non_optional_params_multiple_params_stacked_inverse(0, 1, 2)

    assert len(warn_record) == 3
    messages = ["bla1 param1", "bla2 param2", "bla3 param3"]
    for i, message in enumerate(messages):
        assert message in warn_record[i].message.args[0]
    assert result == (0, 1, 2)

    with pytest.raises(ValueError):
        # duplicate entry across decorators
        @deprecate_parameters("bla1 {parameter}", "param1")
        @deprecate_parameters_if_set("bla2 {parameter}", "param1")
        @deprecate_parameters("bla3 {parameter}", "param3")
        def func_non_optional_params_multiple_params_stacked_inverse_duplicate(
            param1: int, param2: int, param3: int
        ) -> Tuple[int, int, int]:
            return param1, param2, param3


def test_deprecate_parameter_stacked_nested(
    func_non_optional_params_multiple_params_stacked_nested_counterparts,
):
    # check: Only the first stacked deprecator construct in the callstack should raise warnings.
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_non_optional_params_multiple_params_stacked_nested(
            func_non_optional_params_multiple_params_stacked_nested_counterparts,
            0,
            1,
        )

    # ensures, that the second - nested - deprecator construct does not raise warnings
    assert len(warn_record) == 2
    messages = ["bla1 param1", "bla2 param2"]
    for i, message in enumerate(messages):
        assert message in warn_record[i].message.args[0]
    # ensure singleton state has been cleared
    assert _WarningActuator().outermost_deprecator is None
    assert result == (0, 1)


def test_deprecate_parameter_stacked_nested_inverse(
    func_non_optional_params_multiple_params_stacked_nested_counterparts,
):
    # check: Only the first stacked deprecator construct in the callstack should raise warnings.
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_non_optional_params_multiple_params_stacked_nested_inverse(
            func_non_optional_params_multiple_params_stacked_nested_counterparts,
            0,
            1,
        )

    # ensures, that the second - nested - deprecator construct does not raise warnings
    assert len(warn_record) == 2
    messages = ["bla2 param2", "bla1 param1"]
    for i, message in enumerate(messages):
        assert message in warn_record[i].message.args[0]
    # ensure singleton state has been cleared
    assert _WarningActuator().outermost_deprecator is None
    assert result == (0, 1)


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
        def func_test_raise_invalid_parameter_decorator_one_param(param1: int) -> int:
            return param1

    # check: init raises no params exist
    with pytest.raises(ValueError):

        @deprecate_parameters_if_set(MESSAGE, "param1")
        def func_test_raise_invalid_parameter_decorator_no_param() -> None:
            pass

    # check: do NOT warn when deprecated params not specified
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = func_optional_params_multiple_params(0)
    assert result == (0, None, None)

    # check: raise if used without kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_optional_params_multiple_params(0, 1, 2)
    assert len(warn_record) == 2
    assert result == (0, 1, 2)

    # check: raise if used with kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_optional_params_multiple_params(0, param2=1, param3=2)
    assert len(warn_record) == 2
    assert result == (0, 1, 2)

    # check: raise mixed if used without kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_optional_params_mixed_optional(0, 1)
    assert len(warn_record) == 2
    assert result == (0, 1)

    # check: raise mixed if used without kwargs
    with pytest.warns(DeprecationWarning) as warn_record:
        result = func_optional_params_mixed_optional(param1=0, param2=1)
    assert len(warn_record) == 2
    assert result == (0, 1)


def test_default_parameter_deprecation_warning_texts_generic_version(
    default_parameter_deprecation_texts_generic: str,
):
    @deprecate_parameters_if_set(default_parameter_deprecation_texts_generic, "param1")
    def func_test_raise_unparametrized_decorator(param1: int) -> int:
        return param1

    with pytest.warns(DeprecationWarning) as warn_record:
        func_test_raise_unparametrized_decorator(1)
    assert len(warn_record) == 1
    assert "param1" in warn_record[0].message.args[0]


def test_default_deprecation_warning_texts_specific_version(
    default_param_deprecation_texts_specific: Callable,
):
    # accounts for different amounts of version parameters
    n_params = len(
        inspect.signature(default_param_deprecation_texts_specific).parameters.keys()
    )
    strings = [f"1.0.{n}" for n in range(n_params)]

    @deprecate_parameters_if_set(
        default_param_deprecation_texts_specific(*strings), "param1"
    )
    def func_test_raise_unparametrized_decorator(param1: int) -> int:
        return param1

    with pytest.warns(DeprecationWarning) as warn_record:
        func_test_raise_unparametrized_decorator(1)

    assert len(warn_record) == 1
    message = warn_record[0].message.args[0]
    assert "param1" in message
    for string in strings:
        assert string in message


def test_default_function_deprecation_warning_texts_generic_version(
    default_function_deprecation_texts_generic: str,
):
    assert "{function}" in default_function_deprecation_texts_generic


def test_generic_deprecation_warning(default_deprecation_texts_generic: str):
    assert type(default_deprecation_texts_generic) == str


@pytest.fixture(
    scope="class",
    params=[False, True],
)
def is_test_optional_parameters(request) -> bool:
    return request.param


def test_get_specific_function_deprecation_warning(
    is_test_optional_parameters: bool,
    default_function_deprecation_texts_specific: Callable,
):

    if not is_test_optional_parameters:
        message = default_function_deprecation_texts_specific(
            function_name="test_func", deprecated_in="1.0.0"
        )
    else:
        # accounts for different amounts of version parameters
        n_params = (
            len(
                inspect.signature(
                    default_param_deprecation_texts_specific
                ).parameters.keys()
            )
            - 2
        )
        test_strings = [f"test_{n}" for n in range(n_params)]
        message = default_function_deprecation_texts_specific(
            function_name="test_func", deprecated_in="1.0.0", *test_strings
        )
        for string in test_strings:
            assert string in message

    assert "test_func" in message
    assert "1.0.0" in message
