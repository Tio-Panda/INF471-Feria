import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
# from PIL import Image

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from engine import Engine
from shapes import Polygon, ShapeMap

class EngineSAM(Engine):
    def __init__(self, engine, **kwargs):

        # Configurar el device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"using device: {device}")

        if device.type == "cuda":
            # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        elif device.type == "mps":
            print(

                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )

        # Obtener los pesos y config de SAM2.1
        sam2_checkpoint = "../checkpoint/sam2.1_hiera_tiny.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

        # Construir el modelo
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
        kwargs['model'] = kwargs.get('model', sam2)
        
        self.mask_generator = SAM2AutomaticMaskGenerator(**kwargs)
    
    def get_shape_map(self, img):
        img = np.array(img)
        anns = self.mask_generator.generate(img)
        anns = sorted(anns, key=lambda x: x['area'], reverse=True)

        polys = []
        for aux in anns:
            mask = aux["segmentation"].astype(bool).astype(np.uint8) * 255
            area = aux["area"]
            bbox = np.array(aux["bbox"]).flatten().astype(int)

            polys.append(Polygon(mask, area, bbox))

        shape_map = ShapeMap(polys)

        return shape_map
