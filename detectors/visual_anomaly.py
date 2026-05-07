import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torchvision.transforms as T

from game_qa.config import (
    ANOMALY_IMG_SIZE,
    ANOMALY_THRESHOLD,
    AUTOENCODER_EPOCHS,
    AUTOENCODER_BATCH,
    MODEL_DIR,
)

logger = logging.getLogger(__name__)

_transform = T.Compose([
    T.Resize(ANOMALY_IMG_SIZE),
    T.ToTensor(),           # [0,1] float
])

# model                                                                
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64x64
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16x16
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, stride=2, padding=1), # 8x8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))



# detector                                                             
class VisualAnomalyDetector:
    MODEL_PATH = MODEL_DIR / "autoencoder.pt"

    def __init__(self, threshold: float = ANOMALY_THRESHOLD):
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvAutoencoder().to(self.device)
        self.trained = False
        self._recon_errors: list[float] = []   # rolling history for adaptive threshold

    # TRAINING

    def _frames_to_tensor(self, frames: list[np.ndarray]) -> torch.Tensor:
        tensors = []
        for f in frames:
            pil = Image.fromarray(f.astype(np.uint8))
            tensors.append(_transform(pil))
        return torch.stack(tensors)

    def train(self, normal_frames: list[np.ndarray], epochs: int = AUTOENCODER_EPOCHS):
        if len(normal_frames) < 10:
            logger.warning("Too few frames (%d) to train autoencoder reliably", len(normal_frames))

        X = self._frames_to_tensor(normal_frames).to(self.device)
        dataset = TensorDataset(X)
        loader = DataLoader(dataset, batch_size=AUTOENCODER_BATCH, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = criterion(recon, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            if (epoch + 1) % 5 == 0:
                logger.info("Autoencoder epoch %d/%d  loss=%.5f", epoch + 1, epochs, avg)

        torch.save(self.model.state_dict(), self.MODEL_PATH)
        self.trained = True
        logger.info("Autoencoder trained and saved → %s", self.MODEL_PATH)

    def load(self):
        if self.MODEL_PATH.exists():
            self.model.load_state_dict(
                torch.load(self.MODEL_PATH, map_location=self.device)
            )
            self.trained = True
            logger.info("Autoencoder loaded from %s", self.MODEL_PATH)
        else:
            logger.warning("No saved model at %s", self.MODEL_PATH)

    # INFERENCE

    def score(self, frame: np.ndarray) -> float:
        """Return per-pixel MSE reconstruction error (higher = more anomalous)."""
        self.model.eval()
        with torch.no_grad():
            pil = Image.fromarray(frame.astype(np.uint8))
            x = _transform(pil).unsqueeze(0).to(self.device)
            recon = self.model(x)
            mse = nn.functional.mse_loss(recon, x).item()
        self._recon_errors.append(mse)
        return mse

    def detect(self, frame: np.ndarray) -> tuple[bool, float]:        
        mse = self.score(frame)

        # adaptive threshold:
        #  max(fixed_threshold, mean + 3*std of recent scores)
        threshold = self.threshold
        if len(self._recon_errors) > 30:
            history = np.array(self._recon_errors[-100:])
            adaptive = history.mean() + 3 * history.std()
            threshold = max(self.threshold, adaptive)

        # returns (is_anomaly, score)
        return mse > threshold, mse

    def visualize_reconstruction(
        self, frame: np.ndarray, save_path: Optional[Path] = None
    ) -> np.ndarray:
        import cv2

        self.model.eval()
        with torch.no_grad():
            pil = Image.fromarray(frame.astype(np.uint8))
            x = _transform(pil).unsqueeze(0).to(self.device)
            recon = self.model(x).squeeze(0).cpu().permute(1, 2, 0).numpy()

        orig_resized = np.array(pil.resize(ANOMALY_IMG_SIZE)) / 255.0
        recon_clipped = np.clip(recon, 0, 1)
        diff = np.abs(orig_resized - recon_clipped)

        def to_u8(arr):
            return (arr * 255).astype(np.uint8)

        composite = np.hstack([to_u8(orig_resized), to_u8(recon_clipped), to_u8(diff * 5)])

        if save_path:
            Image.fromarray(composite).save(save_path)

        return composite

class FrameDiffDetector:

    def __init__(self, diff_threshold: float = 0.25):
        self.diff_threshold = diff_threshold  # fraction of pixels that changed significantly
        self._prev_frame: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        # returns (is_anomaly, changed_pixel_fraction).
        small = np.array(Image.fromarray(frame).resize((64, 64))).astype(np.float32) / 255.0

        if self._prev_frame is None:
            self._prev_frame = small
            return False, 0.0

        diff = np.abs(small - self._prev_frame)
        changed = (diff > 0.15).any(axis=-1).mean()
        self._prev_frame = small

        return changed > self.diff_threshold, float(changed)
