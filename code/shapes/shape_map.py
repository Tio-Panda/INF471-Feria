from PIL import Image
from itertools import chain

class ShapeMap():
    def __init__(self, poly_list):
        self.poly_list = poly_list

    def __getitem__(self, idx):
        return self.poly_list[idx]

    def __len__(self):
        return len(self.poly_list)
    
    def set_map_sub_polygons(self, n_max, l_high, l_low, range=(0, -1)):
        start, end = range
        for poly in self.poly_list[start:end]:
            poly.set_sub_polygons(n_max, l_high, l_low)

    def get_concatenated_sub_polygons(self):
        polys = []
        for poly in self.poly_list:
            if poly.sub_polygons != None:
                polys = [*polys, *poly.sub_polygons]
        return polys
    
    def get_map_img(self, range=(0, -1)):
        if not self.poly_list:
            return None

        h, w = self.poly_list[0].mask.shape

        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        start, end = range
        for poly in self.poly_list[start:end]:
            canvas = Image.alpha_composite(canvas, poly.get_img())

        return canvas
    
    def get_sub_polygons_img(self, range=(0, -1)):
        h, w = self.poly_list[0].mask.shape

        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        start, end = range
        for poly in self.poly_list[start:end]:
            canvas = Image.alpha_composite(canvas, poly.get_img_sub_polygons())

        return canvas

    

    
    
    
