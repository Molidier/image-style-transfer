## Image Style Transfer using PyTorch

### Overview
This project implements an image style transfer application using PyTorch. It combines the style of one image with the content of another by optimizing a deep learning model based on a pretrained VGG network. The project also includes quantization for performance optimization.

### Features
1. **Style Transfer**:
   - Applies the artistic style of one image to the content of another using a modified VGG-19 architecture.
   - Supports GPU acceleration for faster computation.
   
2. **Quantization**:
   - Implements static and dynamic quantization techniques for reduced model size and inference latency.
   - Saves quantized models for deployment.

3. **Visualization**:
   - Displays intermediate and final results using Matplotlib.

### Dependencies
- Python 3.8 or higher
- PyTorch 2.x
- torchvision
- Pillow
- Matplotlib

### Getting Started

#### 1. Clone the repository:
```bash
git clone https://github.com/your_username/image-style-transfer.git
cd image-style-transfer
```

#### 2. Install dependencies:
```bash
pip install torch torchvision matplotlib pillow
```

#### 3. Run Style Transfer:
```bash
python style_transfer.py
```

#### 4. Quantize the Model:
```bash
python quantize.py
```

### Files
- **`style_transfer.py`**: Contains the implementation of the style transfer algorithm.
- **`quantize.py`**: Provides static and dynamic quantization functionalities for the model.

### Outputs
- **Transformed Images**: View results in a Matplotlib window or save them as needed.
- **Models**: Checkpoints for both the unquantized and quantized models are saved in the `models/` directory.

### Notes
- Ensure that style and content images are of the same dimensions.
- Modify hyperparameters like `num_steps`, `style_weight`, and `content_weight` in `style_transfer.py` to experiment with results.
