import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import timm
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import io
import base64

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="RadVision AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS - Dark clinical aesthetic
# =====================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #050A0F;
    --bg-secondary: #0A1520;
    --bg-card: #0D1B2A;
    --bg-card-hover: #112236;
    --accent-cyan: #00E5FF;
    --accent-green: #00FF9D;
    --accent-amber: #FFB800;
    --accent-red: #FF4D6D;
    --text-primary: #E8F4FD;
    --text-secondary: #7A9DB8;
    --text-muted: #3D5A73;
    --border-subtle: #1A3048;
    --border-accent: rgba(0, 229, 255, 0.3);
    --glow-cyan: 0 0 20px rgba(0, 229, 255, 0.15);
    --glow-green: 0 0 20px rgba(0, 255, 157, 0.15);
}

* { box-sizing: border-box; }

.stApp {
    background: var(--bg-primary);
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(0, 229, 255, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 100%, rgba(0, 255, 157, 0.03) 0%, transparent 50%);
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* ---- HEADER ---- */
.rad-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.8rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
}
.rad-logo-group { display: flex; align-items: center; gap: 1rem; }
.rad-logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-green));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: var(--glow-cyan);
}
.rad-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    line-height: 1;
}
.rad-subtitle {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 3px;
    font-family: 'Space Mono', monospace;
}
.rad-badge {
    background: rgba(0, 229, 255, 0.08);
    border: 1px solid rgba(0, 229, 255, 0.2);
    border-radius: 20px;
    padding: 0.3rem 0.9rem;
    font-size: 0.7rem;
    color: var(--accent-cyan);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}

/* ---- UPLOAD ZONE ---- */
.upload-wrapper {
    background: var(--bg-card);
    border: 1.5px dashed var(--border-accent);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.upload-wrapper::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,229,255,0.03) 0%, transparent 70%);
    pointer-events: none;
}

/* ---- CARDS ---- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0, 229, 255, 0.2); }

.metric-label {
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.4rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--accent-cyan);
}

/* ---- SECTION HEADERS ---- */
.section-head {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem;
}
.section-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-cyan);
    box-shadow: 0 0 8px var(--accent-cyan);
    flex-shrink: 0;
}
.section-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text-secondary);
}

/* ---- FINDINGS BOX ---- */
.findings-box {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-left: 3px solid var(--accent-green);
    border-radius: 0 12px 12px 0;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--glow-green);
}
.findings-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-green);
    margin-bottom: 0.6rem;
}
.findings-text {
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text-primary);
}

.impression-box {
    background: linear-gradient(135deg, rgba(0, 229, 255, 0.05), rgba(0, 255, 157, 0.03));
    border: 1px solid rgba(0, 229, 255, 0.25);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--glow-cyan);
}
.impression-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 0.6rem;
}
.impression-text {
    font-size: 1rem;
    line-height: 1.7;
    color: var(--text-primary);
    font-weight: 500;
}

/* ---- PROCESSING STEPS ---- */
.step-item {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.85rem;
    color: var(--text-secondary);
}
.step-icon { font-size: 1rem; width: 1.5rem; text-align: center; }
.step-pending { color: var(--text-muted); }
.step-active { color: var(--accent-amber); }
.step-done { color: var(--accent-green); }

/* ---- STREAMLIT OVERRIDES ---- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), #0099CC) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(0, 229, 255, 0.25) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(0, 229, 255, 0.4) !important;
}

div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed rgba(0, 229, 255, 0.3) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

/* ---- DIVIDER ---- */
.rad-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 1.5rem 0;
}

/* ---- STATUS PILL ---- */
.status-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(0, 255, 157, 0.08);
    border: 1px solid rgba(0, 255, 157, 0.2);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.68rem;
    color: var(--accent-green);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'Space Mono', monospace;
}
.pulse-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.7); }
}

/* ---- HEATMAP LABEL ---- */
.view-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    margin-top: 0.5rem;
}

/* ---- SCAN LINE ANIMATION ---- */
.scan-anim {
    position: relative;
    overflow: hidden;
}
.scan-anim::after {
    content: '';
    position: absolute;
    left: 0; top: -100%;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    animation: scanline 2s linear infinite;
}
@keyframes scanline {
    0% { top: -2px; }
    100% { top: 100%; }
}

/* Adjust st.image */
[data-testid="stImage"] img {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CONFIG
# =====================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "razent/SciFive-base-Pubmed_PMC"
VIT_MODEL = "vit_tiny_patch16_224"
MODEL_PATH = "./vlm_model/vlm_epoch_3.pt"
MAX_LEN = 64

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# =====================================================
# MODEL DEFINITION
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
# GRAD-CAM
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
        tokens = self.model.vit.forward_features(input_tensor)
        cls_token = tokens[:, 0]
        projected = self.model.projection(cls_token).unsqueeze(1)
        decoder_input_ids = torch.ones((1, 1), dtype=torch.long).to(DEVICE)
        outputs = self.model.lm(
            inputs_embeds=projected,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        score = outputs.logits.mean()
        self.model.zero_grad()
        score.backward()

        grads = self.gradients[0].detach().cpu()
        acts = self.activations[0].detach().cpu()
        grads = grads[1:]
        acts = acts[1:]
        weights = grads.mean(dim=0)
        cam = torch.matmul(acts, weights)
        cam = cam.reshape(14, 14).numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam

# =====================================================
# CACHED MODEL LOADER
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = VisionLanguageModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, tokenizer

# =====================================================
# INFERENCE
# =====================================================
def run_inference(image: Image.Image, model, tokenizer):
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Generate text
    with torch.no_grad():
        embeddings = model(img_tensor)
        outputs = model.lm.generate(
            inputs_embeds=embeddings,
            max_length=MAX_LEN,
            num_beams=4,
            early_stopping=True
        )
    impression = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # GradCAM
    gradcam = ViTGradCAM(model)
    cam = gradcam.generate(img_tensor)

    img_np = np.array(image.resize((224, 224))) / 255.0

    # Smooth & threshold
    cam_smooth = cv2.GaussianBlur(cam, (21, 21), 0)
    cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() + 1e-8)
    threshold = np.percentile(cam_smooth, 85)
    cam_mask = np.zeros_like(cam_smooth)
    cam_mask[cam_smooth >= threshold] = cam_smooth[cam_smooth >= threshold]

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(float) / 255
    overlay = np.clip(img_np * 0.7 + heatmap * 0.6, 0, 1)

    return impression, img_np, cam, cam_mask, overlay

def numpy_to_pil(arr):
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def cam_to_pil(cam):
    colored = cm.jet(cam)[:, :, :3]
    return numpy_to_pil(colored)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="rad-header">
    <div class="rad-logo-group">
        <div class="rad-logo-icon">🫁</div>
        <div>
            <div class="rad-title">RadVision AI</div>
            <div class="rad-subtitle">Chest X-Ray Intelligence Platform</div>
        </div>
    </div>
    <div style="display:flex;gap:0.8rem;align-items:center;">
        <div class="rad-badge">VLM v3.0</div>
        <div class="rad-badge">ViT + SciFive</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# LAYOUT
# =====================================================
left_col, mid_col, right_col = st.columns([1.1, 1.4, 1.4], gap="large")

# =====================================================
# LEFT — Upload + Model Info
# =====================================================
with left_col:
    st.markdown("""
    <div class="section-head">
        <div class="section-dot"></div>
        <div class="section-title">Input Image</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop your chest X-ray here",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div style='margin-top:0.8rem;'>
            <div class="metric-card">
                <div class="metric-label">File Name</div>
                <div style='font-size:0.8rem;color:#7A9DB8;font-family:Space Mono,monospace;word-break:break-all;'>{uploaded.name}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Dimensions</div>
                <div class="metric-value">{image.width} × {image.height}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mode</div>
                <div class="metric-value" style='font-size:1rem;'>{image.mode}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0;'>
            <div style='font-size:3rem;margin-bottom:0.8rem;opacity:0.3;'>🩻</div>
            <div style='color:#3D5A73;font-size:0.82rem;line-height:1.6;'>
                Upload a chest X-ray image<br>to begin AI analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Architecture info
    st.markdown("""
    <div class="section-head" style='margin-top:0.5rem;'>
        <div class="section-dot" style='background:#FFB800;box-shadow:0 0 8px #FFB800;'></div>
        <div class="section-title">Model Architecture</div>
    </div>
    <div style='font-size:0.78rem;color:#7A9DB8;line-height:1.9;'>
        <span style='color:#FFB800;font-family:Space Mono,monospace;'>ENCODER</span> &nbsp; ViT-Tiny / 16×16 patches<br>
        <span style='color:#FFB800;font-family:Space Mono,monospace;'>BRIDGE &nbsp;</span> &nbsp; Linear Projection 192→768<br>
        <span style='color:#FFB800;font-family:Space Mono,monospace;'>DECODER</span> &nbsp; SciFive (PubMed+PMC)<br>
        <span style='color:#FFB800;font-family:Space Mono,monospace;'>XAI &nbsp;&nbsp;&nbsp;&nbsp;</span> &nbsp; Grad-CAM on ViT Block[-1]
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    analyze_btn = st.button("🔬  Analyze X-Ray", disabled=uploaded is None)

# =====================================================
# MID — Processing Steps + Heatmaps
# =====================================================
with mid_col:
    st.markdown("""
    <div class="section-head">
        <div class="section-dot" style='background:#FFB800;box-shadow:0 0 8px #FFB800;'></div>
        <div class="section-title">Analysis Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    steps_placeholder = st.empty()

    STEPS = [
        ("🖼️", "Image preprocessing & normalization"),
        ("🔭", "ViT patch tokenization (16×16)"),
        ("🧠", "Visual feature extraction"),
        ("🔗", "Cross-modal projection (192→768)"),
        ("📡", "SciFive language generation"),
        ("🌡️", "Grad-CAM attention mapping"),
        ("✅", "Report compilation"),
    ]

    def render_steps(done=0, active=-1):
        html = "<div style='margin-bottom:1rem;'>"
        for i, (icon, label) in enumerate(STEPS):
            if i < done:
                cls = "step-done"
                tag = f'<span style="color:#00FF9D;font-size:0.7rem;margin-left:auto;font-family:Space Mono,monospace;">DONE</span>'
            elif i == active:
                cls = "step-active"
                tag = f'<span style="color:#FFB800;font-size:0.7rem;margin-left:auto;font-family:Space Mono,monospace;">●</span>'
            else:
                cls = "step-pending"
                tag = ""
            html += f"""
            <div class="step-item {cls}">
                <span class="step-icon">{icon}</span>
                <span>{label}</span>
                {tag}
            </div>"""
        html += "</div>"
        return html

    steps_placeholder.html(render_steps())

    heatmap_placeholder = st.empty()
    heatmap_status = st.empty()

    if not uploaded:
        heatmap_placeholder.html("""
        <div style='background:#0D1B2A;border:1px solid #1A3048;border-radius:12px;
                    padding:3.5rem;text-align:center;margin-top:1rem;'>
            <div style='font-size:2.5rem;opacity:0.2;margin-bottom:0.8rem;'>🌡️</div>
            <div style='color:#3D5A73;font-size:0.8rem;'>GradCAM visualization<br>will appear here</div>
        </div>
        """)

# =====================================================
# RIGHT — Results
# =====================================================
with right_col:
    st.markdown("""
    <div class="section-head">
        <div class="section-dot" style='background:#00FF9D;box-shadow:0 0 8px #00FF9D;'></div>
        <div class="section-title">Radiology Report</div>
    </div>
    """, unsafe_allow_html=True)

    results_placeholder = st.empty()

    if not uploaded:
        results_placeholder.html("""
        <div style='background:#0D1B2A;border:1px solid #1A3048;border-radius:12px;
                    padding:3.5rem;text-align:center;'>
            <div style='font-size:2.5rem;opacity:0.2;margin-bottom:0.8rem;'>📋</div>
            <div style='color:#3D5A73;font-size:0.8rem;'>AI-generated findings &<br>impression will appear here</div>
        </div>
        """)

# =====================================================
# ANALYSIS EXECUTION
# =====================================================
if analyze_btn and uploaded:
    with mid_col:
        # Animate pipeline steps
        for step_i in range(len(STEPS)):
            steps_placeholder.html(render_steps(done=step_i, active=step_i))
            time.sleep(0.38)

        steps_placeholder.html(render_steps(done=len(STEPS)))

        heatmap_placeholder.html("""
        <div class="scan-anim" style='background:#0D1B2A;border:1px solid rgba(0,229,255,0.2);
                    border-radius:12px;padding:2rem;text-align:center;margin-top:0.5rem;'>
            <div style='color:#00E5FF;font-family:Space Mono,monospace;font-size:0.75rem;
                        letter-spacing:0.15em;'>GENERATING ATTENTION MAP...</div>
        </div>
        """)

    # Load model & run
    with st.spinner(""):
        try:
            model, tokenizer = load_model()
            impression, img_np, cam_raw, cam_mask, overlay = run_inference(image, model, tokenizer)

            # Split impression into findings + summary (heuristic)
            sentences = [s.strip() for s in impression.replace(". ", ".\n").split("\n") if s.strip()]
            if len(sentences) > 1:
                findings_text = " ".join(sentences[:-1])
                impression_text = sentences[-1]
            else:
                findings_text = impression
                impression_text = impression

            # Show heatmaps
            with mid_col:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(numpy_to_pil(overlay), use_container_width=True)
                    st.markdown('<div class="view-label">GradCAM Overlay</div>', unsafe_allow_html=True)
                with col_b:
                    st.image(cam_to_pil(cam_mask), use_container_width=True)
                    st.markdown('<div class="view-label">Attention Heatmap</div>', unsafe_allow_html=True)

                heatmap_status.html("""
                <div style='margin-top:0.8rem;'>
                    <span class="status-pill">
                        <span class="pulse-dot"></span>
                        Analysis complete
                    </span>
                </div>
                """)

            # Show report
            with right_col:
                results_placeholder.html(f"""
                <div>


                    <div class="impression-box">
                        <div class="impression-title">🩺 Impression</div>
                        <div class="impression-text">{impression_text}</div>
                    </div>

                    <div class="rad-divider"></div>

                    <div style='display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.5rem;'>
                        <div style='background:rgba(0,255,157,0.06);border:1px solid rgba(0,255,157,0.15);
                                    border-radius:8px;padding:0.6rem 1rem;flex:1;text-align:center;'>
                            <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                                        color:#3D5A73;letter-spacing:0.15em;text-transform:uppercase;'>Model</div>
                            <div style='font-size:0.78rem;color:#00FF9D;margin-top:0.2rem;'>SciFive PMC</div>
                        </div>
                        <div style='background:rgba(0,229,255,0.06);border:1px solid rgba(0,229,255,0.15);
                                    border-radius:8px;padding:0.6rem 1rem;flex:1;text-align:center;'>
                            <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                                        color:#3D5A73;letter-spacing:0.15em;text-transform:uppercase;'>Device</div>
                            <div style='font-size:0.78rem;color:#00E5FF;margin-top:0.2rem;'>{DEVICE.upper()}</div>
                        </div>
                        <div style='background:rgba(255,184,0,0.06);border:1px solid rgba(255,184,0,0.15);
                                    border-radius:8px;padding:0.6rem 1rem;flex:1;text-align:center;'>
                            <div style='font-family:Space Mono,monospace;font-size:0.6rem;
                                        color:#3D5A73;letter-spacing:0.15em;text-transform:uppercase;'>XAI</div>
                            <div style='font-size:0.78rem;color:#FFB800;margin-top:0.2rem;'>Grad-CAM</div>
                        </div>
                    </div>

                    <div style='margin-top:1.2rem;background:rgba(255,77,109,0.05);border:1px solid rgba(255,77,109,0.15);
                                border-radius:8px;padding:0.8rem 1rem;'>
                        <div style='font-size:0.7rem;color:#FF4D6D;font-family:Space Mono,monospace;
                                    letter-spacing:0.1em;'>⚠️ RESEARCH USE ONLY</div>
                        <div style='font-size:0.72rem;color:#3D5A73;margin-top:0.3rem;line-height:1.5;'>
                            This AI output is not a substitute for professional radiological diagnosis.
                            Always consult a licensed radiologist for clinical decisions.
                        </div>
                    </div>
                </div>
                """)

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.info("Make sure your model checkpoint is at: `./vlm_model/vlm_epoch_3.pt`")