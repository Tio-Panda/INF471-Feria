import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageChops
import cv2

EXTENSIONS = [".jpg", ".jpeg", ".png"]

def get_imgs_array(path):
    imgs: list[np.ndarray] = []
    for root, _, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in EXTENSIONS):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert("RGB")
                imgs.append(np.array(img))
    return imgs

def get_imgs_PIL_array(path):
    imgs = []
    for root, _, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in EXTENSIONS):
                img_path = os.path.join(root, file)
                img = Image.open(img_path).convert("RGB")
                imgs.append(img)
    return imgs

def plot_imgs_tuple_array(imgs_tuples, title_format:str, height=3, width=3):
    n = len(imgs_tuples)

    cols = math.ceil(math.sqrt(n))

    rows = math.ceil(n / cols)

    figsize_width = width * cols
    figsize_height = height * rows
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_width, figsize_height))

    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()


    for idx, img_tuple in enumerate(imgs_tuples):
        ax = axes[idx]
        ax.imshow(img_tuple[1])
        ax.axis("off")
        ax.set_title(title_format.format(img_tuple[0]))

    for j in range(n, rows * cols):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def get_image_from_solution(poly_img_list, canvas):
    N, M = canvas.get_canvas().size
    initial_solution_img = Image.new("RGBA", (N,M), (0, 0, 0, 0))

    for img in poly_img_list:
        initial_solution_img.paste(img, (0,0), img)

    return initial_solution_img

def grid_masks_flat(h, w, rows, cols):
    if rows <= 0 or cols <= 0:
        raise ValueError("rows y cols deben ser > 0")

    # Bordes enteros de cada celda (límites inclusivos)
    r_edges = np.linspace(0, h, rows + 1, dtype=int)
    c_edges = np.linspace(0, w, cols + 1, dtype=int)

    masks = np.zeros((rows * cols, h, w), dtype=bool)

    k = 0
    for r in range(rows):
        r0, r1 = r_edges[r], r_edges[r + 1]
        for c in range(cols):
            c0, c1 = c_edges[c], c_edges[c + 1]
            masks[k, r0:r1, c0:c1] = True
            k += 1
    return masks

def get_img_with_bg(img, bg, color=(255, 255, 255, 255)):
    w, h = img.size
    
    out = Image.new("RGBA", (w,h), color)
    out = Image.alpha_composite(out, bg)
    out = Image.alpha_composite(out, img)

    return out