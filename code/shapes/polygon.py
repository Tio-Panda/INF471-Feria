import numpy as np
from PIL import Image
import random

from shapes import get_n_subdivsions_by_area, split_by_watershed


class Polygon:
    def __init__(self, mask, area, bbox):
        self.mask = mask
        self.area = area
        self.x_min, self.y_min, self.bbox_w, self.bbox_h = bbox
        self.sub_polygons = None

    def set_sub_polygons(self, n_max, l_high, l_low):
        n = get_n_subdivsions_by_area(self.area, l_high, l_low, n_max)

        self.sub_polygons = []
        sub_masks = split_by_watershed(self.mask, n)
        for s_mask in sub_masks:
            ys, xs = np.where(s_mask)
            if ys.size == 0:
                raise ValueError("La máscara está vacía")

            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()

            bbox_w = x_max - x_min + 1
            bbox_h = y_max - y_min + 1

            bbox = [int(x_min), int(y_min), int(bbox_w), int(bbox_h)]

            s_mask = s_mask.astype(bool).astype(np.uint8) * 255
            self.sub_polygons.append(Polygon(s_mask, -1, bbox))

    def get_img(self):
        h, w = self.mask.shape[0:2]
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        rgb = (np.random.rand(3) * 255).astype(np.uint8)
        alpha = int(255 * 0.5)

        color = tuple(rgb.tolist()) + (alpha,)

        layer = Image.new("RGBA", (w, h), color)

        mask_img = Image.fromarray((self.mask * alpha).astype(np.uint8), mode="L")
        layer.putalpha(mask_img)

        img = Image.alpha_composite(img, layer)

        return img
    
    def get_img_sub_polygons(self):
        if self.sub_polygons != None:
            h, w = self.mask.shape[0:2]
            img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

            for poly in self.sub_polygons:
                rgb = (np.random.rand(3) * 255).astype(np.uint8)
                alpha = int(255 * 0.5)

                color = tuple(rgb.tolist()) + (alpha,)

                layer = Image.new("RGBA", (w, h), color)

                mask_img = Image.fromarray((poly.mask * alpha).astype(np.uint8), mode="L")
                layer.putalpha(mask_img)

                img = Image.alpha_composite(img, layer)

            return img
        else:
            return self.get_img
            
    def get_img_with_texture(self, canvas, density_palette):
        N, M = self.mask.shape
        img = Image.new("RGBA", (M,N), (0, 0, 0, 0))

        density = density_palette.get_img_density(canvas.get_masked_img(self.mask))
        patch = density_palette[density]

        patch = patch.resize((self.bbox_w, self.bbox_h), Image.Resampling.LANCZOS)
        img.paste(patch, (self.x_min, self.y_min))
        alpha = Image.fromarray(self.mask.astype(np.uint8), mode="L")
        img.putalpha(alpha)

        return img