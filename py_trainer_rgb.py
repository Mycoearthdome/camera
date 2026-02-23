import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

# ---------------- Config ----------------
W, H = 640, 480
PIXELS = W*H
FRAME_FILE = "/dev/shm/nn_frames.bin"
DEVICE = "cuda"

# ---------------- RGB Autoencoder ----------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Net().to(DEVICE)
opt = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

# ---------------- Frame Reader ----------------
def read_frame():
    if not os.path.exists(FRAME_FILE):
        return None
    data = np.fromfile(FRAME_FILE, dtype=np.float32)
    if data.size < PIXELS*3: # 3 channels
        return None
    frame = data[-PIXELS*3:].reshape(1,3,H,W)
    return torch.from_numpy(frame).float().to(DEVICE)

# ---------------- Training Loop ----------------
step = 0
print("Trainer started")
while True:
    x = read_frame()
    if x is None:
        time.sleep(0.05)
        continue

    y = model(x)
    loss = loss_fn(y, x)

    opt.zero_grad()
    loss.backward()
    opt.step()

    step += 1

    if step % 50 == 0:
        print(f"step {step} loss={loss.item():.6f}")

    # Export to ONNX every 200 steps
    if step % 200 == 0:
        torch.onnx.export(
            model,
            torch.randn(1,3,H,W).to(DEVICE),
            "model.onnx",
            input_names=["input_0"],
            output_names=["output_0"],
            opset_version=18,
            dynamic_axes={"input_0": {0: "batch"}, "output_0": {0: "batch"}}
        )
        print("ONNX updated")