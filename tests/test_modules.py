import importlib
import pkgutil

import pytest

import kartothek


@pytest.mark.skip("skip this test case until ktk implements __all__")
def test_all(module):
    assert hasattr(module, "__all__")
    assert type(module.__all__) == tuple

    missing_in_module = {
        attr for attr in module.__all__ if getattr(module, attr, None) is None
    }
    assert missing_in_module == set()

    true_members = {
        attr
        for attr in dir(module)
        if (getattr(getattr(module, attr), "__module__", None) == module.__name__)
        and not attr.startswith("_")
    }
    missing_in_all = true_members - set(module.__all__)
    assert missing_in_all == set()

    assert (
        tuple(sorted(module.__all__)) == module.__all__
    ), "module list in __all__ should be sorted"


@pytest.mark.skip("skip this test case until ktk implements __all__")
def pytest_generate_tests(metafunc):
    mod = kartothek
    ids = [mod.__name__]
    objs = [mod]

    # iterate over all sub-modules
    for _importer, name, _ispkg in pkgutil.walk_packages(
        mod.__path__, prefix=(mod.__name__ + ".")
    ):
        # use importlib and NOT the _importer, otherwise Pickle gets really confused
        mod2 = importlib.import_module(name)

        ids.append(name)
        objs.append(mod2)

    metafunc.parametrize(argnames="module", argvalues=objs, ids=ids)
