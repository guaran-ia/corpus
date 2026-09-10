import os
import sys
import yaml
import importlib

class ExperimentLoader(yaml.SafeLoader):
    """Custom YAML loader to import callables dynamically using !function."""
    pass

def function_constructor(loader: type[yaml.SafeLoader], node):
    path = loader.construct_scalar(node)
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

ExperimentLoader.add_constructor("!function", function_constructor)

def load_config(config_path: str) -> dict:
    # 1. Resolve project root (e.g. /disk/pipeline) or current working directory
    # Adding the CWD and parent directory checks ensures sys.path includes the root
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    # 2. Safely load the YAML
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=ExperimentLoader)