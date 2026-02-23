import torch
import torch.nn as nn

W, H = 640, 480

# RGB autoencoder
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid() # Ensure outputs are 0..1 for RGB
        )
    def forward(self, x):
        return self.net(x)

model = Net().cuda()
model.eval() # Set to inference mode

# Dummy input to match RGB camera input
dummy_input = torch.randn(1, 3, H, W).cuda() # batch=1, channels=3

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input_0"],
    output_names=["output_0"],
    opset_version=18,
    dynamic_axes={"input_0": {0: "batch"}, "output_0": {0: "batch"}}
)

print("model.onnx created for RGB output")
