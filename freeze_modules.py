from collections.abc import Iterable


SUPPORTED_MODULE_NAMES = ("encoder",)


def _normalize_requested_modules(freeze_cfg) -> list[str]:
    if not freeze_cfg or not freeze_cfg.get("enabled", False):
        return []

    modules = freeze_cfg.get("modules", [])
    if modules is None:
        return []
    if isinstance(modules, str):
        return [modules]
    if not isinstance(modules, Iterable):
        raise TypeError("freeze.modules must be a string or iterable of strings")
    return list(modules)


def apply_freeze_configuration(model, freeze_cfg) -> list[str]:
    requested = _normalize_requested_modules(freeze_cfg)
    unknown = sorted(set(requested) - set(SUPPORTED_MODULE_NAMES))
    if unknown:
        raise ValueError(
            "unsupported freeze modules: "
            + ", ".join(unknown)
            + f"; supported: {', '.join(SUPPORTED_MODULE_NAMES)}"
        )

    frozen = []
    for name in requested:
        submodule = getattr(model, name, None)
        if submodule is None:
            raise ValueError(f"model is missing freezeable module: {name}")
        for parameter in submodule.parameters():
            parameter.requires_grad_(False)
        frozen.append(name)
    return frozen
