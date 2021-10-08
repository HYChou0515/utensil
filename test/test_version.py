from pathlib import Path
import toml
import utensil


def test_versions_are_in_sync():
    """Checks if the pyproject.toml and utensil.__init__.py
    __version__ are in sync."""

    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with open(str(path), encoding='utf-8') as f:
        pyproject = toml.loads(f.read())
    pyproject_version = pyproject["tool"]["poetry"]["version"]

    package_init_version = utensil.__version__

    assert package_init_version == pyproject_version
