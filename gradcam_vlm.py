import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import timm
from torchvision import transforms
from transformers import AutoModelForSeq2SeqLM
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
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# =====================================================
# MODEL DEFINITION
# =====================================================
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(VIT_MODEL, pretrained=False)
        self.vit.head = nn.Identity()

        self.projection = nn.Linear(192,768)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def forward(self,x):
        tokens = self.vit.forward_features(x)
        return tokens

# Load trained model
model = VisionLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# =====================================================
# GRAD-CAM FOR ViT
# =====================================================
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer = model.vit.blocks[-1].norm1

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):

        # Forward through vision encoder
        tokens = self.model.vit.forward_features(input_tensor)

        # Use CLS token for language alignment
        cls_token = tokens[:,0]

        projected = self.model.projection(cls_token)
        projected = projected.unsqueeze(1)

        # IMPORTANT: T5 requires decoder input
        decoder_input_ids = torch.ones((1,1), dtype=torch.long).to(DEVICE)

        outputs = self.model.lm(
            inputs_embeds=projected,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )

        # Use prediction logits as target
        score = outputs.logits.mean()

        self.model.zero_grad()
        score.backward()

        grads = self.gradients[0].detach().cpu()
        acts = self.activations[0].detach().cpu()

        # Remove CLS token
        grads = grads[1:]
        acts = acts[1:]

        weights = grads.mean(dim=0)
        cam = torch.matmul(acts, weights)

        cam = cam.reshape(14,14).numpy()

        cam = np.maximum(cam,0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam = cv2.resize(cam,(224,224))

        return cam

# =====================================================
# HEATMAP GENERATION (FIXED)
# =====================================================
def generate_heatmap(path):

    image = Image.open(path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    gradcam = ViTGradCAM(model)
    cam = gradcam.generate(img_tensor)

    # Convert image to numpy
    img = np.array(image.resize((224,224))) / 255.0

    # -------------------------------
    # 🔥 NEW PROFESSIONAL VISUALIZATION
    # -------------------------------

    # Smooth CAM to remove patch noise
    cam_smooth = cv2.GaussianBlur(cam, (21,21), 0)

    # Normalize
    cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() + 1e-8)

    # Keep only strongest regions
    threshold = np.percentile(cam_smooth, 85)
    cam_mask = np.zeros_like(cam_smooth)
    cam_mask[cam_smooth >= threshold] = cam_smooth[cam_smooth >= threshold]

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255*cam_mask), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(float)/255

    # Blend overlay
    overlay = img * 0.7 + heatmap * 0.6
    overlay = np.clip(overlay, 0, 1)

    return img, cam, overlay

# =====================================================
# MAIN
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
    plt.title("GradCAM Heatmap")
    plt.imshow(cam,cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.show()