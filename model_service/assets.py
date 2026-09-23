"""Control-plane policy for administrator-provided local model artifact bundles."""
import json
from pathlib import Path

from .contracts import ModelConfig, ServiceError


def check_assets(config: ModelConfig, allowed_roots: list[str]) -> None:
    roots = [Path(p).resolve() for p in allowed_roots]
    paths = [config.path] if config.path else []
    for key, value in config.options.items():
        if key.endswith(("_path", "_dir")) and key != "infer_path" and isinstance(value, str):
            paths.append(value)
    checked = set()
    for value in paths:
        path = Path(value).resolve()
        if not any(path.is_relative_to(root) for root in roots):
            raise ServiceError("path_forbidden", "Model assets must be inside an allowed model root", 403)
        if not path.exists():
            raise ServiceError("model_files_missing", f"Model asset does not exist: {path}", 422)
        # Entry-point files load sibling weights/tokenizers implicitly.
        bundle = path if path.is_dir() else path.parent
        if bundle in checked:
            continue
        checked.add(bundle)
        for child in bundle.rglob("*"):
            if child.is_symlink() and not any(child.resolve().is_relative_to(root) for root in roots):
                raise ServiceError("path_forbidden", "Model bundle contains an escaping symlink", 403)
            if child.name.endswith(".index.json") or child.name in {"tokenizer_config.json", "processor_config.json", "preprocessor_config.json"}:
                try:
                    data = json.loads(child.read_text(encoding="utf-8"))
                except (ValueError, OSError) as exc:
                    raise ServiceError("invalid_model_metadata", "Could not read model artifact metadata", 422) from exc
                _check_references(data, child.parent, roots)


def _check_references(data, parent: Path, roots: list[Path]) -> None:
    if not isinstance(data, dict):
        return
    references = []
    weight_map = data.get("weight_map")
    if isinstance(weight_map, dict):
        references.extend(weight_map.values())
    for key, value in data.items():
        if key.endswith("_file") and isinstance(value, str):
            references.append(value)
        elif key.endswith("_files") and isinstance(value, list):
            references.extend(value)
        elif isinstance(value, dict):
            _check_references(value, parent, roots)
    for reference in references:
        if not isinstance(reference, str) or not reference:
            raise ServiceError("invalid_model_metadata", "Invalid model file reference", 422)
        resolved = (parent / reference).resolve()
        if "://" in reference or not any(resolved.is_relative_to(root) for root in roots):
            raise ServiceError("path_forbidden", "Model metadata references an artifact outside allowed roots", 403)
