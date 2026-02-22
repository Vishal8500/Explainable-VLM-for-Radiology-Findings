import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import timm
from torchvision import transforms
from PIL import Image

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"
VIT_MODEL = "vit_tiny_patch16_224"

MODEL_PATH = "./vlm_model/vlm_epoch_3.pt"   # FINAL TRAINED MODEL
MAX_LEN = 64

# =====================================================
# IMAGE TRANSFORM
# =====================================================
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# =====================================================
# MODEL DEFINITION (same as training)
# =====================================================
class VisionLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = timm.create_model(VIT_MODEL, pretrained=False)
        self.vit.head = nn.Identity()

        self.projection = nn.Linear(192, 768)
        self.lm = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    def forward(self, image):
        vit_features = self.vit(image)
        projected = self.projection(vit_features)
        projected = projected.unsqueeze(1)

        return projected

# =====================================================
# LOAD MODEL
# =====================================================
print("Loading trained VLM model...")

model = VisionLanguageModel().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("✅ Model Loaded Successfully")

# =====================================================
# GENERATION FUNCTION
# =====================================================
def generate_impression(image_path):

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    # Extract visual embeddings
    with torch.no_grad():
        embeddings = model(image_tensor)

        outputs = model.lm.generate(
            inputs_embeds=embeddings,
            max_length=MAX_LEN,
            num_beams=4,
            early_stopping=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text

# =====================================================
# TEST ON SAMPLE IMAGE
# =====================================================
if __name__ == "__main__":

    image_path = input("\nEnter chest X-ray image path: ")

    result = generate_impression(image_path)

    print("\n==============================")
    print("🩺 GENERATED IMPRESSION:")
    print("==============================\n")
    print(result)