import numpy as np
import sam2.sam2_image_predictor
import torch

class Tracker:
    def __init__(self, device: torch.device):
        self.segmenter = sam2.sam2_image_predictor.SAM2ImagePredictor.from_pretrained('facebook/sam2-hiera-base-plus')
        self.device = str(device)

    def identify(self, image, target_info: np.ndarray):
        # with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.segmenter.reset_predictor()
            self.segmenter.set_image(image)
            return self.segmenter.predict(box=target_info, multimask_output=False)

    def track(self, image, previous_logits: np.ndarray):
        # with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=torch.bfloat16):
            self.segmenter.set_image(image)
            return self.segmenter.predict(mask_input=previous_logits, multimask_output=False)