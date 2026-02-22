import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import timm
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"
VIT_MODEL = "vit_tiny_patch16_224"
MODEL_PATH = "./vlm_model/vlm_epoch_3.pt"

# =====================================================
# IMAGE TRANSFORM
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================================================
# MODEL
# =====================================================
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(VIT_MODEL, pretrained=False)
        self.vit.head = nn.Identity()

        self.projection = nn.Linear(192, 768)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def forward(self, x):
        features = self.vit.forward_features(x)
        return features

# =====================================================
# LOAD MODEL
# =====================================================
model = VisionLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================================================
# GRAD CAM CLASS
# =====================================================
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook last transformer block
        self.target_layer = model.vit.blocks[-1].norm1

        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):

        output = self.model(input_tensor)

        # Use mean activation as target
        target = output.mean()

        self.model.zero_grad()
        target.backward()

        grads = self.gradients.cpu().data.numpy()[0]
        acts = self.activations.cpu().data.numpy()[0]

        weights = np.mean(grads, axis=0)

        cam = np.zeros(acts.shape[0], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cam.reshape(14, 14)  # ViT patch grid

        cam = cv2.resize(cam, (224, 224))
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

# =====================================================
# GENERATE HEATMAP
# =====================================================
def generate_heatmap(image_path):

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    gradcam = ViTGradCAM(model)
    cam = gradcam.generate(img_tensor)

    # Convert image to numpy
    img = np.array(image.resize((224, 224))) / 255.0

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap / 255.0

    overlay = heatmap * 0.4 + img

    return img, cam, overlay

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":

    path = input("Enter image path: ")

    img, cam, overlay = generate_heatmap(path)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Grad-CAM")
    plt.imshow(cam, cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()