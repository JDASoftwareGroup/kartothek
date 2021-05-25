from kartothek.core.cube.constants import KTK_CUBE_UUID_SEPERATOR
from kartothek.core.dataset import _validate_uuid


def test_uuid_seperator_valid():
    assert _validate_uuid(KTK_CUBE_UUID_SEPERATOR)
