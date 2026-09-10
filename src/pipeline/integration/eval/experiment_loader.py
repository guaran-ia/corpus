import importlib
import yaml

class ExperimentLoader(yaml.SafeLoader):
    """Custom YAML loader to import callables dynamically using !function."""
    pass

def function_constructor(loader:type[yaml.SafeLoader], node):
    path = loader.construct_scalar(node)
    module_name, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, func_name)

# Register the !function tag
ExperimentLoader.add_constructor("!function", function_constructor)

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=ExperimentLoader)