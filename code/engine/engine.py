import importlib
from abc import ABC, abstractmethod
from typing import Any

class Engine(ABC):
    def __new__(cls, engine, **kwargs):
        if cls is Engine:
            if engine == "SAM":
                m = importlib.import_module("engine.sam")
                EngineSAM = getattr(m, "EngineSAM")
                return super(Engine, EngineSAM).__new__(EngineSAM)
            
            else:
                raise ValueError(f"Engine no definido")

        return super.__new__(cls)

    def __init__(self, engine, **kwargs):
        pass

    @abstractmethod
    def get_shape_map(self, img) -> Any:
        pass
