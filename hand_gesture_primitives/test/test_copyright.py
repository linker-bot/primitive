# Copyright (c) 2024, All rights reserved.
import pytest
from ament_copyright.main import main


@pytest.mark.copyright
def test_copyright():
    rc = main(argv=[".", "test"])
    assert rc == 0, "Found errors"
