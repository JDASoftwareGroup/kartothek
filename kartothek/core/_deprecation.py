"""
Code was taken from pandas

https://github.com/pandas-dev/pandas/blob/c5a47118d73a53a016a28103726b57e27924bf29/pandas/util/_decorators.py#L66

with the following license:

BSD 3-Clause License

Copyright (c) 2008-2012, AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import warnings
from functools import wraps


def deprecate_kwarg(old_arg_name, new_arg_name, mapping=None, stacklevel=2):
    """
    Decorator to deprecate a keyword argument of a function.
    Parameters
    ----------
    old_arg_name : str
        Name of argument in function to deprecate
    new_arg_name : str or None
        Name of preferred argument in function. Use None to raise warning that
        ``old_arg_name`` keyword is deprecated.
    mapping : dict or callable
        If mapping is present, use it to translate old arguments to
        new arguments. A callable must do its own value checking;
        values not found in a dict will be forwarded unchanged.
    Examples
    --------
    The following deprecates 'cols', using 'columns' instead
    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name='columns')
    ... def f(columns=''):
    ...     print(columns)
    ...
    >>> f(columns='should work ok')
    should work ok
    >>> f(cols='should raise warning')
    FutureWarning: cols is deprecated, use columns instead
      warnings.warn(msg, FutureWarning)
    should raise warning
    >>> f(cols='should error', columns="can\'t pass do both")
    TypeError: Can only specify 'cols' or 'columns', not both
    >>> @deprecate_kwarg('old', 'new', {'yes': True, 'no': False})
    ... def f(new=False):
    ...     print('yes!' if new else 'no!')
    ...
    >>> f(old='yes')
    FutureWarning: old='yes' is deprecated, use new=True instead
      warnings.warn(msg, FutureWarning)
    yes!
    To raise a warning that a keyword will be removed entirely in the future
    >>> @deprecate_kwarg(old_arg_name='cols', new_arg_name=None)
    ... def f(cols='', another_param=''):
    ...     print(cols)
    ...
    >>> f(cols='should raise warning')
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    >>> f(another_param='should not raise warning')
    should not raise warning
    >>> f(cols='should raise warning', another_param='')
    FutureWarning: the 'cols' keyword is deprecated and will be removed in a
    future version please takes steps to stop use of 'cols'
    should raise warning
    """

    if mapping is not None and not hasattr(mapping, "get") and not callable(mapping):
        raise TypeError(
            "mapping from old to new argument values " "must be dict or callable!"
        )

    def _deprecate_kwarg(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_arg_value = kwargs.pop(old_arg_name, None)

            if new_arg_name is None and old_arg_value is not None:
                msg = (
                    "the '{old_name}' keyword is deprecated and will be "
                    "removed in a future version. "
                    "Please take steps to stop the use of '{old_name}'"
                ).format(old_name=old_arg_name)
                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                kwargs[old_arg_name] = old_arg_value
                return func(*args, **kwargs)

            if old_arg_value is not None:
                if mapping is not None:
                    if hasattr(mapping, "get"):
                        new_arg_value = mapping.get(old_arg_value, old_arg_value)
                    else:
                        new_arg_value = mapping(old_arg_value)
                    msg = (
                        "the {old_name}={old_val!r} keyword is deprecated, "
                        "use {new_name}={new_val!r} instead"
                    ).format(
                        old_name=old_arg_name,
                        old_val=old_arg_value,
                        new_name=new_arg_name,
                        new_val=new_arg_value,
                    )
                else:
                    new_arg_value = old_arg_value
                    msg = (
                        "the '{old_name}' keyword is deprecated, "
                        "use '{new_name}' instead"
                    ).format(old_name=old_arg_name, new_name=new_arg_name)

                warnings.warn(msg, FutureWarning, stacklevel=stacklevel)
                if kwargs.get(new_arg_name, None) is not None:
                    msg = (
                        "Can only specify '{old_name}' or '{new_name}', " "not both"
                    ).format(old_name=old_arg_name, new_name=new_arg_name)
                    raise TypeError(msg)
                else:
                    kwargs[new_arg_name] = new_arg_value
            return func(*args, **kwargs)

        return wrapper

    return _deprecate_kwarg
