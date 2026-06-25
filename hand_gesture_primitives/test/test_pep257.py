import pytest
from ament_pep257.main import main


@pytest.mark.linter
def test_pep257():
    rc = main(argv=[])
    assert rc == 0, "Found code style errors / warnings"
