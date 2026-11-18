import pytest

from nykteris.synth import generate


@pytest.fixture(scope="session")
def night(tmp_path_factory):
    out = tmp_path_factory.mktemp("night")
    return generate(out, seed=3)


@pytest.fixture(scope="session")
def nina_night(tmp_path_factory):
    out = tmp_path_factory.mktemp("nina")
    return generate(out, seed=4, flavor="nina")
