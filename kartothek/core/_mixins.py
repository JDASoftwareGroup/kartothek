# -*- coding: utf-8 -*-
import collections
import inspect


class CopyMixin:
    def copy(self, **kwargs):
        constructor_args = inspect.signature(self.__init__).parameters
        init_args = collections.OrderedDict()

        for arg in constructor_args.keys():
            if arg == "self":
                continue
            init_args[arg] = kwargs.get(arg, getattr(self, arg, None))

        return type(self)(**init_args)
