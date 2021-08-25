import inspect
import traceback
import warnings
from functools import partial, wraps
from typing import Callable, Tuple

# This constant string is supposed to be passed as message to the decorators in the current migration effort and to be
# subsequently removed.
MULTI_TABLE_FEATURE_DEPRECATION_WARNING = (
    "The `{parameter}` keyword is deprecated and will be removed in the next major"
    "release in an effort to remove the multi table feature."
)


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
    if len(params) < 1:
        raise ValueError(
            "At least one parameter must be specified when using this decorator!"
        )

    if len(params) > len(set(params)):  # check for duplicates
        raise ValueError("Duplicate parameter assignment in decorator definition!")

    arg_spec = inspect.getfullargspec(func)
    if not (
        arg_spec.varargs or arg_spec.varkw
    ):  # deactivate check for decorated functions with *args or **kwargs
        func_args = arg_spec.args
        if not all([parameter in func_args for parameter in params]):
            raise ValueError(
                "Invalid parameter in decorator definition: "
                + ", ".join(sorted(set(params) - set(func_args)))
                + "!"
            )


def deprecate_parameters(warning: str, *parameters: str) -> Callable:
    """
    Decorator, raising warnings that specified parameters of the decorated function are deprecated and will be removed
    or changed in the future.

    .. note:: Please avoid using this decorator with other decorators. The correct call origin can not be passed if this
        decorator is nested inside others. If you absolutely have to use it with other decorators, add it last in order
        to enable it to parse the final function signature.

    Examples
    --------
    >>> from kartothek.utils.migration_helpers import deprecate_parameters
    >>> @deprecate_parameters('Parameter {parameter} is deprecated!', 'param1', 'param2')
    ... def func(param1: str, param2: int, param3: float):
    ...    return param1, param2, param3
    ...
    >>> # Warnings will be generated for `param1` and `param2`
    >>> func('example', 0, 5.0)
    ('example', 0, 5.0)

    Parameters
    ----------
    warning: str
        warning, the DeprecationWarnings will be raised with. Please make sure to include the substring '{parameter}'
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
            for parameter in parameters:
                warnings.warn(
                    # prints the call origin, in order to indicate to the user where the code has to be adjusted.
                    warning.format(parameter=parameter)
                    + "\nCalled in: "
                    + str(traceback.extract_stack()[-2])
                    + "\nAffected Kartothek function: "
                    + func.__name__,
                    DeprecationWarning,
                )
            return func(*args, **kwargs)

        _check_params(func=func, params=parameters)
        return wraps_func

    return wrapper


def deprecate_parameters_if_set(warning, *parameters: str) -> Callable:
    """
    Decorator, raising warnings that specified parameters of the decorated function are deprecated and will be
    removed or changed in the future. This warning is only raised for optional parameters, if the parameter is actually
    set when called in order to avoid confusion and limit the users visibility of the change process they are not
    affected by.

    .. note:: Please avoid using this decorator with other decorators. The correct call origin can not be passed if this
        decorator is nested inside others. If you absolutely have to use it with other decorators, add it last in order
        to enable it to parse the final function signature.

    .. note:: Do not decorate parameters hiding behind \\*args or \\*\\*kwargs!

    Examples
    --------
    >>> from kartothek.utils.migration_helpers import deprecate_parameters_if_set
    >>> message = 'Parameter {parameter} is deprecated!'
    >>> @deprecate_parameters_if_set(message, 'param2', 'param3', 'param4')
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
        warning, the DeprecationWarnings will be raised with. Please make sure to include the substring '{parameter}'
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
        def wraps_func(parameters_positions, *args, **kwargs):
            def raise_warning(parameter):
                warnings.warn(
                    warning.format(parameter=parameter)
                    + "\nCalled in: "
                    + str(traceback.extract_stack()[-3])
                    + "\nAffected Kartothek function: "
                    + func.__name__,
                    DeprecationWarning,
                )

            n_args = len(args)
            # (n_args - 1) because we are comparing a length with an index
            deprecated_arg_params = [
                parameter
                for parameter, position in parameters_positions.items()
                if position <= (n_args - 1)
            ]
            for parameter in deprecated_arg_params:
                raise_warning(parameter)
            for kwarg_key in set(kwargs.keys()) - set(deprecated_arg_params):
                if kwarg_key in parameters_positions.keys():
                    raise_warning(kwarg_key)

            return func(*args, **kwargs)

        _check_params(func, parameters)
        func_args = inspect.getfullargspec(func).args
        parameters_mapping = {  # required for resolving optional parameters being passed as non-kwargs
            parameter: func_args.index(parameter) for parameter in parameters
        }
        return partial(wraps_func, parameters_mapping)

    return wrapper
