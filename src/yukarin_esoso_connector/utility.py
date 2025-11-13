import re

import torch
from upath import UPath


def extract_number(f: UPath) -> int:
    s = re.findall(r"\d+", str(f))
    return int(s[-1]) if s else -1


def get_predictor_model_path(
    model_dir: UPath,
    iteration: int | None = None,
    prefix: str = "predictor_",
    postfix: str = ".pth",
) -> UPath:
    if iteration is None:
        paths = model_dir.glob(prefix + "*" + postfix)
        model_path = list(sorted(paths, key=extract_number))[-1]
    else:
        model_path = model_dir / (prefix + str(iteration) + postfix)
    return model_path


def remove_weight_norm(m: torch.nn.Module) -> None:
    try:
        torch.nn.utils.remove_weight_norm(m)
    except ValueError:
        pass
