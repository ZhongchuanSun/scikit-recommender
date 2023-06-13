from importlib.util import find_spec
from importlib import import_module
from collections import OrderedDict

__all__ = ["ModelRegistry"]


class ModelRegistry:
    def __init__(self):
        self.models = OrderedDict()

    def register_model(self, model_name, model_class):
        self.models[model_name] = model_class

    def load_skrec_model(self, model_name: str, spec_path="skrec.recommender") -> bool:
        spec_path = f"{spec_path}.{model_name}"

        if find_spec(spec_path):
            module = import_module(spec_path)
        else:
            print(f"Module '{spec_path}' is not found.")
            # raise ModuleNotFoundError(f"Module '{spec_path}' is not found.")
            return False

        if hasattr(module, model_name):
            model_class = getattr(module, model_name)
        else:
            print(f"Import {model_name} failed from {module.__file__}!")
            # raise ImportError(f"Import {model_name} failed from {module.__file__}!")
            return False

        self.register_model(model_name, model_class)
        return True

    def get_model(self, model_name: str):
        return self.models.get(model_name, None)

    def list_models(self):
        return list(self.models.keys())
