from joblib import dump, load

class Prefabric():
    def __init__(self, canvas, shape_map):
        self.canvas = canvas
        self.shape_map = shape_map

class PrefabricLoader():
    def __init__(self, path):
        self.path = path

    def save_prefabric(self, canvas, shape_map, name):
        pf = Prefabric(canvas, shape_map)
        dump(pf, f"{self.path}/{name}.joblib")


    def load_prefabric(self, name):
        pf = load(f"{self.path}/{name}.joblib")
        canvas = pf.canvas
        shape_map = pf.shape_map

        return (canvas, shape_map)