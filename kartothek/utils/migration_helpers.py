import inspect
import warnings
from functools import wraps
from typing import Callable, Optional, Tuple

# Parameter deprecation messages: Pass these to the parameter deprecators below
# Inserting the right parameter will be handled automatically.

DEPRECATION_WARNING_REMOVE_PARAMETER = "The `{parameter}` keyword is deprecated and will be removed in an upcoming version."

# Do not use with deprecate_parameters_if_set since the warning should implicitly include default value fallbacks
DEPRECATION_WARNING_PARAMETER_NON_OPTIONAL_GENERIC_VERSION = (
    "The `{parameter}` keyword will be non-Optional in an upcoming version."
)


def get_deprecation_warning_remove_parameter_multi_table(
    deprecated_in: str, removed_in: str
) -> str:
    return (
        "The `{parameter}` keyword is deprecated in version: "
        + deprecated_in
        + " and will be removed in version: "
        + removed_in
        + " in an effort to remove the multi table feature."
    )


def get_deprecation_warning_remove_dict_multi_table(
    deprecated_in: str, changed_in: str
) -> str:
    return (
        "The logic of the`{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be changed in version: "
        + changed_in
        + " to not accept a dict of values anymore in an effort to remove the multi table feature."
    )


def get_deprecation_warning_parameter_non_optional(
    deprecated_in: str, changed_in: str
) -> str:
    return (
        "The `{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be non-Optional in version: "
        + changed_in
        + "."
    )


def get_parameter_replaced_by_deprecation_warning(
    replaced_by: str, deprecated_in: str, changed_in: str
) -> str:
    return (
        "The `{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be replaced by "
        + replaced_by
        + " in version: "
        + changed_in
        + " ."
    )


def get_parameter_default_value_deprecation_warning(
    from_value: str, to_value: str, deprecated_in: str, changed_in: str
) -> str:
    return (
        "The default value of the `{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be changed from "
        + from_value
        + " to "
        + to_value
        + " in version: "
        + changed_in
        + " ."
    )


def get_parameter_type_change_deprecation_warning(
    from_type: str, to_type: str, deprecated_in: str, changed_in: str
) -> str:
    return (
        "The type of the `{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be changed from "
        + from_type
        + " to "
        + to_type
        + " in version: "
        + changed_in
        + " ."
    )


def get_parameter_generic_replacement_deprecation_warning(
    replacing_parameter: str, deprecated_in: str, changed_in: str
) -> str:
    return (
        "The `{parameter}` keyword is deprecated since version: "
        + deprecated_in
        + " and will be replaced by the "
        + replacing_parameter
        + " parameter in version: "
        + changed_in
        + " ."
    )


# /Parameter deprecation messages


# Function deprecation messages: They are supposed to be used stand-alone because no default handling is provided here
DEPRECATION_WARNING_REMOVE_FUNCTION_GENERIC_VERSION = (
    "The `{function}` keyword is deprecated and will be removed."
)


def get_generic_function_deprecation_waring(function_name: str) -> str:
    return DEPRECATION_WARNING_REMOVE_FUNCTION_GENERIC_VERSION.format(
        function=function_name
    )


def get_specific_function_deprecation_warning(
    function_name: str,
    deprecated_in: str,
    removed_in: Optional[str] = None,
    reason: Optional[str] = None,
):
    return (
        f"The `{function_name}` keyword is deprecated since version "
        + deprecated_in
        + " and will be removed"
        + ((" in version " + removed_in + ".") if removed_in is not None else "")
        + ((" Reason: " + reason + ".") if reason is not None else "")
        + "."
    )


def get_specific_function_deprecation_warning_multi_table(
    function_name: str,
    deprecated_in: str,
    removed_in: Optional[str] = None,
):
    return get_specific_function_deprecation_warning(
        function_name=function_name,
        deprecated_in=deprecated_in,
        removed_in=removed_in,
        reason="Part of the Effort to remove the multi table feature",
    )


# /Function deprecation messages


def _check_params(func: Callable, params: Tuple[str, ...]) -> None:
    """
    Applies validators, verifying the declared decorator parameters.

    Validates:

    #. At least one parameter specified;
    #. No duplicate param declaration;
    #. Declared params match underlying function signature.

    Parameters
    ----------
    params: Tuple[str, ...]
        Declared decorator parameters.

    Raises
    ------
    ValueError
        If validation fails.
    """

    def _raise_invalid_param_error_if_applicable(
        params, func_args, is_maybe_duplicate: bool = False
    ):
        if not all([parameter in func_args for parameter in params]):
            raise ValueError(
                "Invalid "
                + (
                    "or duplicate "
                    if is_maybe_duplicate
                    else "" + "parameter in decorator definition: "
                )
                + ", ".join(sorted(set(params) - set(func_args)))
                + "!"
            )

    if len(params) < 1:
        raise ValueError(
            "At least one parameter must be specified when using this decorator!"
        )

    if len(params) > len(set(params)):  # check for duplicates
        raise ValueError("Duplicate parameter assignment in decorator definition!")

    arg_spec = inspect.getfullargspec(func)
    if not hasattr(func, "kartothek_deprecation_decorator_params"):
        if not (arg_spec.varargs or arg_spec.varkw):
            _raise_invalid_param_error_if_applicable(
                params=params, func_args=inspect.signature(func).parameters.keys()
            )
        else:
            raise ValueError(
                "Decorator cannot be applied to function with *args or **kwargs"
            )
    else:
        _raise_invalid_param_error_if_applicable(
            params=params, func_args=func.kartothek_deprecation_decorator_params  # type: ignore
        )


def _get_trace_message(trace: str) -> str:
    return ("\nCalled in: " + trace) if "/kartothek/" not in trace else ""


def _assemble_warning_message(parameter: str, message: str, func_name: str) -> str:
    return (
        message.format(parameter=parameter)
        + "\nAffected Kartothek function: "
        + func_name
    )


def raise_warning(
    parameter: str, warning: str, func_name: str, stacklevel: int
) -> None:
    # gets original trace message if deprecators have been stacked
    warnings.warn(
        _assemble_warning_message(
            parameter=parameter,
            message=warning,
            func_name=func_name,
        ),
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def _make_decorator_stackable(
    wrapper_func: Callable,
    base_func: Callable,
    exclude_parameters: Tuple[str],
) -> Callable:
    """
    Attaches neccessary meta info directly to the decorator function's objects making multiple instance of these
    deprecation decorators stackable while the parameter check before runtime stays intact.
    This now also facilitates the Blocking of deprecation warnings in the call hierarchy below another annotated
    function in order to prevent the raising of deprecation warnings triggered by kartothek internal calls.

    Parameters
    ----------
    wraps_func
        The deprecation decorator's wraps func, that has the first deprecation decprator in the stacked strucure
        attached as attribute `outermost_stacked_kartothek_deprecator`.
    base_func
        The fuction decorated by the individual deprecation decorator. Please note, that this can either be the
        decorated fuction or another nested decorator.
    exclude_parameters
        Tuple of parameter names, that have been handled by other deprecation decorators already.

    Returns
    -------
    any
        Returns the result of `func(*args, **kwargs)`.
    """
    if hasattr(base_func, "kartothek_deprecation_decorator_params"):
        wrapper_func.kartothek_deprecation_decorator_params = tuple(  # type: ignore
            param
            for param in base_func.kartothek_deprecation_decorator_params  # type: ignore
            if param not in exclude_parameters
        )
    else:
        wrapper_func.kartothek_deprecation_decorator_params = tuple(  # type: ignore
            param
            for param in inspect.signature(base_func).parameters.keys()
            if param not in exclude_parameters
        )

    # Facilitate detection of outermost deprecation warning in order to limit warning reporting of Kartothek internal
    # calls.
    if not hasattr(wrapper_func, "outermost_stacked_kartothek_deprecator"):
        wrapper_func.outermost_stacked_kartothek_deprecator = wrapper_func  # type: ignore
    base_func.outermost_stacked_kartothek_deprecator = (  # type: ignore
        wrapper_func.outermost_stacked_kartothek_deprecator  # type: ignore
    )

    # explicitly preserve signature, facilitating compatibility with other decorators.
    wrapper_func.__signature__ = inspect.signature(base_func)  # type: ignore
    return wrapper_func


class _Singleton(object):
    """
    Official singleton implementation mentioned in the official python documentation at:
    https://www.python.org/download/releases/2.2/descrintro/#__new__
    """

    def __new__(cls, *args, **kwds):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwds)
        return it

    def init(self, *args, **kwds):
        pass


class _WarningActuator(_Singleton):
    """
    _WarningActuator is a sinlgeton, used in doer to save and release a state, blocking deprecator decorator warnings,
    nested inside a call hierarchy, triggered by a decorated function.
    """

    # called on first instantiation
    def init(self):
        self.outermost_deprecator = None

    # called on every instantiation
    def __init__(self):
        pass


def _handle_suppress_warnings_in_subsequent_deprecators(
    wraps_func: Callable, warning_func: Callable, func: Callable, args, kwargs
):
    """
    This function ensures that only the first (stacked) deprecator in the callstack raises warnings in order to suppress
    most deprecation warnings triggered by Kartothek intern calls.

    Parameters
    ----------
    wraps_func
        The deprecation decorator's wraps func, that has the first deprecation decprator in the stacked strucure
        attached as attribute `outermost_stacked_kartothek_deprecator`.
    warning_func
        Function or partial without parameters returning the warning message.
    func
        The fuction decorated by the individual deprecation decorator. Please note, that this can either be the
        decorated fuction or another nested decorator.
    args
        The args that the param:``func`` will be called with.
    kwargs
        The args that the param:``func`` will be called with.

    Returns
    -------
    any
        Returns the result of `func(*args, **kwargs)`.
    """
    actuator = _WarningActuator()

    if actuator.outermost_deprecator is None:
        actuator.outermost_deprecator = (
            wraps_func.outermost_stacked_kartothek_deprecator  # type: ignore
        )

    # Suppresses subsequent deprecation warnings in order to avoid printing numerous warnings in
    # kartothek-triggered function calls.
    if actuator.outermost_deprecator is wraps_func.outermost_stacked_kartothek_deprecator:  # type: ignore
        warning_func()
    try:
        value = func(*args, **kwargs)
    # Ensure singleton state is reset on exception occurring.
    finally:
        if actuator.outermost_deprecator is wraps_func:
            actuator.outermost_deprecator = None
    return value


def deprecate_parameters(warning: str, *parameters: str) -> Callable:
    """
    Decorator, raising warnings that specified parameters of the decorated function are deprecated and will be removed
    or changed in the future.

    .. note:: Please only use this decorator before other decorators preserving the function __name__ and __signature__.
        And note, that the correct call origin can not be returned if this decorator is nested inside others. If you
        absolutely have to use it with other decorators, best add it last.

    ..note:: You may stack `deprecate_parameters` and `deprecate_parameters_if_set` decorators interchanigibly.

    Examples
    --------
    >>> from kartothek.utils.migration_helpers import deprecate_parameters
    >>> message = 'Parameter {parameter} is deprecated due to reason X!'
    >>> message2 = 'Parameter {parameter} is deprecated due to reason Y!'
    >>> @deprecate_parameters(message, 'param1', 'param2')
    ... @deprecate_parameters(message2, 'param4')
    ... def func(param1: str, param2: int, param3: float, param4: float):
    ...    return param1, param2, param3, param4
    ...
    >>> # Warnings will be generated for `param1`, `param2` and `param4` with a different message
    >>> func('example', 0, 5.0, 10.0)
    ('example', 0, 5.0, 10.0)

    Parameters
    ----------
    warning: str
        Warning, the DeprecationWarnings will be raised with. Please make sure to include the substring '{parameter}'
        that will be replaced by the parameter name in the warning.
    *parameters: Tuple [str]
        Tuple of strings denoting the parameters to be marked deprecated.

    Raises
    ------
    DeprecationWarning
        One deprecation warning per parameter containing the formatted passed `warning` string.
    ValueError
        If the validation routines in plave for the decorator are not passed.
        Possible issues:
            No param specified;
            Duplicate param definition;
            Declared param does not match underlying function signature.
    """

    def wrapper(func):
        @wraps(func)
        def wraps_func(*args, **kwargs):
            def warn_logic() -> None:
                for parameter in parameters:
                    raise_warning(
                        parameter=parameter,
                        warning=warning,
                        func_name=func.__name__,
                        stacklevel=5,
                    )

            return _handle_suppress_warnings_in_subsequent_deprecators(
                wraps_func=wraps_func,
                warning_func=warn_logic,
                func=func,
                args=args,
                kwargs=kwargs,
            )

        _check_params(func=func, params=parameters)
        return _make_decorator_stackable(
            wrapper_func=wraps_func, base_func=func, exclude_parameters=parameters
        )

    return wrapper


def deprecate_parameters_if_set(warning, *parameters: str) -> Callable:
    """
    Decorator, raising warnings that specified parameters of the decorated function are deprecated and will be
    removed or changed in the future. This warning is only raised for optional parameters, if the parameter is actually
    set when called in order to avoid confusion and limit the users visibility of the change process they are not
    affected by.

    .. note:: Please only use this decorator before other decorators preserving the function __name__ and __signature__.
        And note, that the correct call origin can not be returned if this decorator is nested inside others. If you
        absolutely have to use it with other decorators, best add it last.

    .. note:: Do not decorate parameters hiding behind \\*args or \\*\\*kwargs!

    ..note:: You may stack `deprecate_parameters` and `deprecate_parameters_if_set` decorators interchanigibly.

    Examples
    --------
    >>> from kartothek.utils.migration_helpers import deprecate_parameters_if_set
    >>> message = 'Parameter {parameter} is deprecated due to reason X!'
    >>> message2 = 'Parameter {parameter} is deprecated due to reason Y!'
    >>> @deprecate_parameters_if_set(message, 'param2', 'param3')
    ... @deprecate_parameters_if_set(message2, 'param4')
    ... def func(param1: str, param2: int, param3: float=None, param4: float=None):
    ...     return param1, param2, param3, param4
    ...
    >>> # Deprecation warnings for parameters: `param2` and `param3`. No warning for `param4` since `param4` has not
    >>> # been specified in the function call.
    >>> func('example', 0, param3=5.0)
    ('example', 0, 5.0, None)

    Parameters
    ----------
    warning: str
        Warning, the DeprecationWarnings will be raised with. Please make sure to include the substring '{parameter}'
        that will be replaced by the parameter name in the warning.
    *parameters: str
        Tuple of strings denoting the parameters to be marked deprecated.

    Raises
    ------
    DeprecationWarning
        One deprecation warning per set parameter containing the formatted passed `warning` string.
    ValueError
        If the validation routines in place for the decorator are not passed.
        Possible issues:
        1) No param specified;
        2) Duplicate param definition;
        3) Declared param does not match underlying function signature.
    """

    def wrapper(func):
        @wraps(func)
        def wraps_func(*args, **kwargs):
            def warn_logic() -> None:
                n_args = len(args)
                # (n_args - 1) because we are comparing a length with an index
                deprecated_arg_params = [
                    parameter
                    for parameter, position in parameters_mapping.items()
                    if position <= (n_args - 1)
                ]
                for parameter in deprecated_arg_params:
                    raise_warning(
                        parameter=parameter,
                        warning=warning,
                        func_name=func.__name__,
                        stacklevel=5,
                    )
                for kwarg_key in set(kwargs.keys()) - set(deprecated_arg_params):
                    if kwarg_key in parameters_mapping.keys():
                        raise_warning(
                            parameter=kwarg_key,
                            warning=warning,
                            func_name=func.__name__,
                            stacklevel=5,
                        )

            return _handle_suppress_warnings_in_subsequent_deprecators(
                wraps_func=wraps_func,
                warning_func=warn_logic,
                func=func,
                args=args,
                kwargs=kwargs,
            )

        _check_params(func=func, params=parameters)

        # decorator is stacked on another deprecation decorator if hasattr==True
        # else decorator is directly applied to the decorated function.
        func_args = (
            func.kartothek_deprecation_decorator_params
            if hasattr(func, "kartothek_deprecation_decorator_params")
            else list(inspect.signature(func).parameters.keys())
        )

        parameters_mapping = (
            {  # required for resolving optional parameters being passed as non-kwargs
                parameter: func_args.index(parameter) for parameter in parameters
            }
        )
        # _make_decorator_stackable required in order to be able to stack multiple of these decorators.
        return _make_decorator_stackable(
            wrapper_func=wraps_func, base_func=func, exclude_parameters=parameters
        )

    return wrapper
