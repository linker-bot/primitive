"""Offline object mesh directory for vision primitives (override via MESH_MODEL_DIR)."""

import os


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def default_mesh_model_dir() -> str:
    env = os.environ.get("MESH_MODEL_DIR", "").strip()
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(_repo_root(), "data", "mesh_model")
