# -*- coding: utf-8 -*-


import inspect
import logging

_logger = logging.getLogger(__name__)


class CopyMixin(object):
    def copy(self, **kwargs):
        constructor_args = inspect.getargspec(self.__init__).args
        init_args = {}
        # first arg is always `self`
        for arg in constructor_args[1:]:
            init_args[arg] = kwargs.get(arg, getattr(self, arg, None))

        return type(self)(**init_args)
