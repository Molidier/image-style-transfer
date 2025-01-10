import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
import os

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model with QuantStub and DeQuantStub for static quantization
class StyleTransferModelQuantized(nn.Module):
    def __init__(self):
        super(StyleTransferModelQuantized, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.layers(x)
        x = self.dequant(x)
        return x

# Load the model
model = StyleTransferModelQuantized().to(device)
model_path = "models/style_transfer_model.pth"

try:
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# -----------------------
# Dynamic Quantization
# -----------------------
try:
    dynamic_quantized_model = torch.quantization.quantize_dynamic(
        model,  # The full-precision model
        {nn.Linear, nn.Conv2d},  # Specify layers to quantize
        dtype=torch.qint8  # Quantized data type
    )
    print("Model dynamically quantized to qint8 successfully.")
    dynamic_quantized_model_path = "models/quantized_style_transfer_model_dynamic_qint8.pth"
    torch.save(dynamic_quantized_model.state_dict(), dynamic_quantized_model_path)
    print(f"Dynamically quantized model saved as '{dynamic_quantized_model_path}'.")
except Exception as e:
    print(f"Dynamic quantization failed: {e}")
