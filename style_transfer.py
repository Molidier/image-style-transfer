import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision.models import vgg19_bn
from torchvision.models import VGG19_BN_Weights

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image processing transformations
loader = transforms.Compose([
    transforms.Resize((360, 640)),  # Resize to 640x630
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalization for VGG
])

def image_loader(image_name):
    """Load an image and preprocess it for the model."""
    image = Image.open(image_name).convert('RGB')  # Ensure 3-channel RGB
    image = loader(image).unsqueeze(0)  # Add batch dimension
    print(f"Loaded image size: {image.size()}")  # Debug output for size
    return image.to(device, torch.float)

# Load style and content images
style_img = image_loader("./data/style/Van_Gogh_resized.jpg")
content_img = image_loader("./data/content_resized/13.jpg")
assert style_img.size() == content_img.size(), "Style and content images must be the same size."

# Convert tensors to PIL images
unloader = transforms.ToPILImage()

# Unnormalize for visualization
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    tensor = tensor * std + mean  # Unnormalize
    return torch.clamp(tensor, 0, 1)

def imshow(tensor, title=None):
    """Display a tensor as an image."""
    image = tensor.cpu().clone()
    image = unnormalize(image).squeeze(0)  # Remove batch dimension after unnormalizing
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.pause(0.001)

# Display the images
plt.figure()
imshow(style_img, title="Style Image")

plt.figure()
imshow(content_img, title="Content Image")

# Ensure the ContentLoss and StyleLoss classes work on `device`
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0, device=device)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    """Compute the Gram matrix for style loss."""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0, device=device)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Normalization layer
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# Load VGG model with updated weights usage
cnn = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1).features.eval()

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    # Clone the CNN and add normalization layer
    model = nn.Sequential(normalization)

    i = 0  # Incremental index for conv layers
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn{i}"
        else:
            continue

        model.add_module(name, layer)

        # Add style loss
        if name in {'conv1', 'conv2', 'conv3', 'conv4', 'conv5'}:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        # Add content loss
        if name == 'conv4':
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

    # Trim the model to avoid unnecessary layers
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (StyleLoss, ContentLoss)):
            break
    model = model[:i + 1]

    return model, style_losses, content_losses

# Run style transfer
input_img = content_img.clone()
plt.figure()
imshow(input_img, title="Input Image")

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=100000, content_weight=10):
    print("Building the style transfer model..")
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean,
                                                                     normalization_std,
                                                                     style_img,
                                                                     content_img)
    
    # Ensure input_img has requires_grad=True
    input_img = input_img.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([input_img], lr=0.01)

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = torch.sum(torch.stack([sl.loss for sl in style_losses]))
            content_score = torch.sum(torch.stack([cl.loss for cl in content_losses]))

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)
        run[0] += 1

    with torch.no_grad():
        input_img.clamp_(0, 1)
    
    torch.save(model.state_dict(), "models/style_transfer_model.pth")

    return input_img

# Execute style transfer
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title="Output Image")
plt.ioff()
plt.show()
