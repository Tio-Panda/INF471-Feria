import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
from numpy.random import default_rng

def get_bg_letter(N, M, canvas, font_size=5, gamma=1):
    SIZE = (N, M)

    FONTS = {
        0: "../fonts/RobotoMono-Bold.ttf",        # 700
        1: "../fonts/RobotoMono-SemiBold.ttf",    # 600
        2: "../fonts/RobotoMono-Medium.ttf",      # 500
        3: "../fonts/RobotoMono-Regular.ttf",     # 400
        4: "../fonts/RobotoMono-Light.ttf",       # 300
        5: "../fonts/RobotoMono-ExtraLight.ttf",  # 200
        6: "../fonts/RobotoMono-Thin.ttf",        # 100
    }

    LOREM = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
            "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
            "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
            "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
            "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
            "deserunt mollit anim id est laborum.").replace(" ", "")

    img = canvas.get_canvas().convert("L")
    if gamma != 1.0:
        lut = [pow(i / 255.0, gamma) * 255 for i in range(256)]
        img = img.point(lut)
    img = ImageOps.autocontrast(img, cutoff=2)

    font_probe = ImageFont.truetype(next(iter(FONTS.values())), font_size)
    CELL_W, CELL_H = font_probe.getbbox("M")[2:]

    cols, rows = SIZE[0] // CELL_W, SIZE[1] // CELL_H
    img_small   = img.resize((cols, rows), Image.Resampling.BILINEAR)
    px          = np.array(img_small)

    LEVELS = len(FONTS)
    bins   = np.linspace(0, 255, LEVELS + 1)
    levels = np.digitize(px, bins) -1 
    levels = np.clip(levels, 0, LEVELS - 1)
    fonts = {lvl: ImageFont.truetype(path, font_size) for lvl, path in FONTS.items()}

    canvas_h = rows * CELL_H                       
    __canvas   = Image.new("RGBA", (SIZE[0], canvas_h), (0,0,0,0))
    draw     = ImageDraw.Draw(__canvas)

    k = 0
    for y in range(rows):
        for x in range(cols):
            lvl   = int(levels[y, x])
            ch    = LOREM[k % len(LOREM)]
            k    += 1
            draw.text(
                (x * CELL_W, y * CELL_H),
                ch,
                font=fonts[lvl],
                fill="black"
            )

    bg_letters = __canvas.resize(SIZE, Image.Resampling.NEAREST)

    return bg_letters


def get_img_details(img, s_umbral, v_umbral):
    img = ImageOps.equalize(img.convert("RGB"))
    img = img.convert("HSV")

    _, s, v = img.split()
   
    s_mask = s.point(lambda p: 255 if p >= s_umbral else 0)
    v_mask = v.point(lambda p: 255 if p >= v_umbral else 0)
    alpha_mask = ImageChops.lighter(s_mask, v_mask)

    output = img.convert("RGBA")
    output.putalpha(alpha_mask)

    return output

def row_fusion(img1, img2, n_stripes, p, seed=11):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    out  = np.empty_like(arr1)
    rng  = default_rng(seed)

    height = arr1.shape[0]
    stripes = np.array_split(np.arange(height), n_stripes)

    for stripe in stripes:
        if stripe.size == 0:
            continue
        if rng.random() < p:
            out[stripe, ...] = arr1[stripe, ...]
        else:
            out[stripe, ...] = arr2[stripe, ...]

    return Image.fromarray(out, mode=img1.mode)

def col_fusion(img1, img2, n_stripes, p, seed=11):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    out  = np.empty_like(arr1)
    rng  = default_rng(seed)

    width = arr1.shape[1]
    stripes = np.array_split(np.arange(width), n_stripes)

    for stripe in stripes:
        if stripe.size == 0:
            continue
        if rng.random() < p:
            out[:, stripe] = arr1[:, stripe]
        else:
            out[:, stripe] = arr2[:, stripe]

    return Image.fromarray(out, mode=img1.mode)

def diagonal_fusion(img1, img2, n_stripes, angle, p, seed=11):
    img1 = img1.convert("RGB")
    img2 = img2.convert("RGB")
    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    h, w = arr1.shape[:2]
    theta = np.deg2rad(angle)
    dx, dy = np.cos(theta), np.sin(theta)
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    proj = x * dx + y * dy
    minp, maxp = proj.min(), proj.max()
    stripes = ((proj - minp) / (maxp - minp) * n_stripes).astype(int)
    stripes = np.clip(stripes, 0, n_stripes - 1)
    rng = default_rng(seed)
    sel = rng.random(n_stripes) < p
    mask = sel[stripes]
    out = np.where(mask[:, :, None], arr1, arr2)
    return Image.fromarray(out, mode=img1.mode)

