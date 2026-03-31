from collections.abc import Mapping
from pathlib import Path


def resolve_warm_start_checkpoint_path(stablewm_home: Path, checkpoint) -> Path:
    if checkpoint in ("", None):
        raise ValueError("warm-start checkpoint must be provided when warm-start is enabled")

    path = Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = Path(stablewm_home).expanduser() / path
    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(f"warm-start checkpoint does not exist: {path}")
    return path


def extract_state_dict(payload):
    if isinstance(payload, Mapping):
        nested = payload.get("state_dict")
        if isinstance(nested, Mapping):
            return nested
        return payload

    state_dict_fn = getattr(payload, "state_dict", None)
    if callable(state_dict_fn):
        return state_dict_fn()

    raise TypeError(f"checkpoint payload does not contain a usable state_dict: {type(payload)!r}")


def _default_loader(path: Path, map_location="cpu", weights_only=True):
    import torch

    return torch.load(path, map_location=map_location, weights_only=weights_only)


def _shape_tuple(value):
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(shape)


def filter_compatible_state_dict(module, state_dict):
    module_state = module.state_dict()
    compatible = {}

    for key, value in state_dict.items():
        if key not in module_state:
            continue
        module_shape = _shape_tuple(module_state[key])
        value_shape = _shape_tuple(value)
        if module_shape is not None and value_shape is not None and module_shape != value_shape:
            continue
        compatible[key] = value

    return compatible


def apply_warm_start(module, checkpoint_path: Path, loader=None, strict: bool = True):
    loader = loader or _default_loader
    payload = loader(Path(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(payload)
    if not strict:
        state_dict = filter_compatible_state_dict(module, state_dict)
    return module.load_state_dict(state_dict, strict=strict)
