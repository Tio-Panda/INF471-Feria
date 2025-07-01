import numpy as np
from PIL import Image
from img_handlers.utils import get_imgs_PIL_array, plot_imgs_tuple_array, grid_masks_flat

class DensityPalette():
    def __init__(self, simple_path, full_path, n_rows, n_cols, levels):
        self.rng = rng = np.random.default_rng()
        self.levels = levels

        # Inicializar la paleta
        self.palette = {}

        for i in range(0,levels):
            self.palette[i] = []
        
        # Clasificar las imagenes por densidad de negro
        simple_imgs_array = get_imgs_PIL_array(simple_path)
        full_imgs_array = get_imgs_PIL_array(full_path)

        # Separar las texturas completas en simples
        for texture_img in full_imgs_array:
            w, h = texture_img.size

            masks = grid_masks_flat(h, w, n_rows, n_cols)

            for mask in masks:
                img = texture_img.convert("RGBA")
                alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
                img.putalpha(alpha)

                ys, xs = np.where(mask)
                if ys.size == 0:
                    return None

                top, bottom = ys.min(), ys.max()
                left, right = xs.min(), xs.max()

                simple_imgs_array.append(img.crop((left, top, right + 1, bottom + 1)))

        for img in simple_imgs_array:
            img = img.convert("L")
            density = self.get_img_density(img)
            self.palette[density].append(img)

        for key, arr in self.palette.items():
            print(f"Densidad {key}: {len(arr)}")


    def __getitem__(self, idx):
        arr = self.palette[idx]
        idx = self.rng.integers(0, len(arr))

        img = arr[idx].rotate(
            np.random.choice([0, 90, 180, 270]),
            resample=Image.Resampling.BICUBIC,
            expand=True,
        ).convert("RGBA")
        return img

    def show_palette(self):
        arr = [(key, img)
            for key, img_array in self.palette.items()
            for img in img_array
        ]
        plot_imgs_tuple_array(arr, "Densidad: {}", height=2, width=2)


    def get_img_density(self, img):
        img = img.convert("RGBA")
        lum = np.array(img.convert("L"), dtype=np.uint8)
        alpha = np.array(img.getchannel("A"), dtype=np.uint8)

        step = 255 / (self.levels - 1)
        lut = np.clip(np.rint(np.arange(256) / step), 0, self.levels - 1).astype(np.uint8)
        lvl = lut[lum]

        w = alpha.astype(np.float32) / 255

        hist = np.bincount(lvl.ravel(), weights=w.ravel(), minlength=self.levels).astype(np.float64)

        total_w = hist.sum()
        if total_w == 0:
            return 0

        sum_wl = (hist * np.arange(self.levels)).sum()
        return int(round(sum_wl / total_w))


    def get_img_density1(self, img):
        img = img.convert("RGBA")
        lum = img.convert("L")
        alpha = img.getchannel("A")

        step = 255 / (self.levels - 1)

        hist = [0.0] * self.levels
        total_w = 0.0
        sum_wl = 0.0

        for g, a in zip(lum.getdata(), alpha.getdata()):
            if a == 0:
                continue
            w = a / 255.0
            lvl = int(round(g / step))
            lvl = min(max(lvl, 0), self.levels - 1)
            hist[lvl] += w
            total_w   += w
            sum_wl    += w * lvl

        if total_w == 0:
            return 0
        
        return int(round(sum_wl / total_w))