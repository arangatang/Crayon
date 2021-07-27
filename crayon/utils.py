from pathlib import Path


def write_to_file(data, path):
    """
    stores data to path and creates any dirs on the way.
    """
    raise NotImplementedError


def crayon_dir():
    path = Path.home() / "Documents" / "crayon"
    path.mkdir(parents=True, exist_ok=True)
    return path


def crayon_results():
    path = crayon_dir() / "results.yml"
    path.touch(exist_ok=True)
    return path
