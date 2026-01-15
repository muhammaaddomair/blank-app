import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

        if not output.requires_grad:
            output.requires_grad_(True)

        output.register_hook(self._grad_hook)

    def _grad_hook(self, grad):
        self.gradients = grad

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam[0].detach().cpu().numpy()

def overlay_cam_on_image(rgb_img, cam, alpha=0.45):
    h, w = rgb_img.shape[:2]
    cam = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (rgb_img * (1 - alpha) + heatmap * alpha).astype(np.uint8)
