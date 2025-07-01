import numpy as np
import lpips
from skimage.metrics import structural_similarity as ssim

loss_fn = lpips.LPIPS(net='alex')

def eval_function(original, img, w=0.5):
    original = original.convert("RGB")
    img = img.convert("RGB")
    ssim_value = ssim(np.array(original.convert("L")), np.array(img.convert("L")))

    _original = np.array(original).astype(np.float32) / 255.0
    _original = _original * 2.0 - 1.0
    _original = lpips.im2tensor(_original)

    _img = np.array(img).astype(np.float32) / 255.0
    _img = _img * 2.0 - 1.0
    _img = lpips.im2tensor(_img)

    lpips_distance = loss_fn.forward(_original, _img).item()

    eval_value = (ssim_value * (1 - w)) + ((1 - lpips_distance) * w)
    return eval_value



