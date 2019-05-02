# -*- coding: utf-8 -*-

import timeit


class AsvBenchmarkConfig(object):
    """
    A mixin class to define default configurations for our benchmarks
    """

    # Use a timer which meassured Wall time (asv default meassures CPU time)
    timer = timeit.default_timer
