import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageChops

from img_handlers.utils import grid_masks_flat
from img_handlers.details import get_bg_letter

class MainCanvas():
    def __init__(self, img, N=1280, M=720):
        TARGET_LANDSCAPE = (N, M)
        TARGET_PORTRAIT  = (M, N)

        img = img.convert("RGB")
        w, h = img.size

        target = TARGET_LANDSCAPE if w >= h else TARGET_PORTRAIT

        result = ImageOps.fit(
            img,
            target,
            method=Image.LANCZOS,
            centering=(0.5, 0.5)
        )
        
        self.img = result
        self.h, self.w = h, w

        self.bg_letter = None
        self.bg_result = None


    def get_canvas(self):
        return self.img

    def get_masked_img(self, mask):
        img = self.img
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(img, (0,0))
        layer.putalpha(Image.fromarray(mask))

        return layer

    def get_masked_letter_img(self, mask):
        img = self.bg_letter
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        layer.paste(img, (0,0))
        layer.putalpha(Image.fromarray(mask))

        return layer
    
    def colorize_img(self, img, factor=0.2):
        out = Image.blend(img, self.get_canvas().convert("RGBA"), factor)
        return out
    
    def set_bg_letter(self, path, font_size=5, gamma=1):
        w, h = self.img.size
        self.bg_letter = get_bg_letter(path, w, h, self, font_size, gamma)

    def set_bg_grid(self, rows, cols, density_palette):
        w, h = self.img.size
        masks = grid_masks_flat(h, w, rows, cols)

        result = Image.new("RGBA", (w, h), (0, 0, 0, 0))

        for mask in masks:
            ys, xs = np.where(mask)
            if ys.size == 0:
                continue

            r0, r1 = ys.min(), ys.max()
            c0, c1 = xs.min(), xs.max()
            bbox_h, bbox_w = r1 - r0 + 1, c1 - c0 + 1

            masked_img = self.get_masked_img(mask)
            density = density_palette.get_img_density(masked_img)
            patch = density_palette[density].resize((bbox_w, bbox_h), Image.Resampling.LANCZOS)

            alpha_arr  = (mask[r0:r1+1, c0:c1+1].astype(np.uint8) * 255)
            patch.putalpha(Image.fromarray(alpha_arr, mode="L"))

            result.paste(patch, (c0, r0), patch)

        self.bg_grid = result
