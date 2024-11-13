import cv2
import numpy as np
import torch

from depth_pro import create_model_and_transforms


class DepthEstimator:

    def __init__(self, device: torch.device):
        self.depth_pro, self.depth_pro_transform = create_model_and_transforms(device=device, precision=torch.half)
        self.depth_pro.eval()

    def predict_depth(self, image: np.ndarray) -> (float, float):
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        input_data = self.depth_pro_transform(image)
        prediction = self.depth_pro.infer(input_data)

        depth = prediction["depth"].detach().cpu().numpy().squeeze()
        focal_length_px = prediction["focallength_px"].detach().cpu().item()
        return depth, focal_length_px

    def _evaluate(self, image: np.ndarray, depth: np.ndarray):
        from matplotlib import pyplot as plt
        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
                max_invdepth_vizu - min_invdepth_vizu
        )

        plt.ion()
        fig = plt.figure()
        ax_rgb = fig.add_subplot(312)
        ax_disp = fig.add_subplot(313)
        ax_actual = fig.add_subplot(311)
        ax_rgb.imshow(image)
        ax_disp.imshow(inverse_depth_normalized, cmap="twilight")
        ax_actual.imshow(depth, cmap='flag', vmin=0.01, vmax=4)
        plt.show(block=True)