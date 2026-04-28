import os
import cv2
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UNet(nn.Module):
    """U-Net architecture for brain tumor segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = nn.Conv2d(n_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512, 3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Conv2d(512, 256, 3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = nn.Conv2d(256, 128, 3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv10 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(self.pool(x1)))
        x3 = F.relu(self.conv3(self.pool(x2)))
        x4 = F.relu(self.conv4(self.pool(x3)))
        x5 = F.relu(self.conv5(self.pool(x4)))

        x6 = F.relu(self.conv6(torch.cat([x4, F.relu(self.upconv1(x5))], dim=1)))
        x7 = F.relu(self.conv7(torch.cat([x3, F.relu(self.upconv2(x6))], dim=1)))
        x8 = F.relu(self.conv8(torch.cat([x2, F.relu(self.upconv3(x7))], dim=1)))
        x9 = F.relu(self.conv9(torch.cat([x1, F.relu(self.upconv4(x8))], dim=1)))
        return self.conv10(x9)


class BrainTumorSegmentation:
    """Brain tumor segmentation using a trained U-Net model."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = DEVICE
        self.model = self._load_model()

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Brain tumor model not found at {self.model_path}. "
                "Download a pre-trained brain tumor segmentation model (.pth) and place it at that path."
            )
            return None
        try:
            model = UNet(n_channels=3, n_classes=1).to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device)
            state_dict = checkpoint.get("state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.eval()
            logger.info(f"Brain tumor model loaded from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading brain tumor model: {e}")
            return None

    def _overlay_mask(self, img: np.ndarray, mask: np.ndarray, output_path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            mask_stacked = np.stack((mask,) * 3, axis=-1)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.axis("off")
            ax.imshow(img)
            ax.imshow(mask_stacked, alpha=0.4, cmap="Reds")
            plt.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            return True
        except Exception as e:
            logger.error(f"Error generating brain tumor overlay: {e}")
            return False

    def predict(self, image_path: str, output_path: str) -> bool:
        if self.model is None:
            logger.error("Brain tumor model is not loaded. Cannot run inference.")
            return False
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img_resized = cv2.resize(img, (256, 256))
            img_tensor = torch.Tensor(img_resized).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            with torch.no_grad():
                mask = self.model(img_tensor).squeeze().cpu().numpy()

            mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
            return self._overlay_mask(img, mask_resized, output_path)
        except Exception as e:
            logger.error(f"Brain tumor segmentation error: {e}")
            return False
