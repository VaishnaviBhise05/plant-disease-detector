"""
Plant Disease Detection - Website-style Streamlit app
Run:  streamlit run app.py
Needs plant_model.keras and class_names.json (created by train.py)
"""
import base64
import io
import json
import os

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="LeafCare AI", page_icon="🌿", layout="wide")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1150px; }

.hero {
    background: linear-gradient(135deg, #14532d 0%, #16a34a 55%, #4ade80 100%);
    border-radius: 24px; padding: 48px 40px; color: #ffffff;
    box-shadow: 0 10px 30px rgba(22,101,52,.35); margin-bottom: 26px;
}
.hero h1 { font-size: 2.8rem; margin: 0 0 8px 0; font-weight: 700; color: #fff; }
.hero p  { font-size: 1.1rem; opacity: .95; margin: 0; max-width: 640px; color: #fff; }
.pill {
    display: inline-block; background: rgba(255,255,255,.2);
    padding: 5px 14px; border-radius: 999px; font-size: .8rem; margin: 18px 8px 0 0;
}

.stat-row { display: flex; gap: 16px; margin-bottom: 26px; flex-wrap: wrap; }
.stat {
    flex: 1; min-width: 150px; background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 16px; padding: 18px; text-align: center;
}
.stat b { display: block; font-size: 1.7rem; color: #166534; }
.stat span { color: #3f6212; font-size: .85rem; }

.card {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 18px;
    padding: 22px; box-shadow: 0 4px 14px rgba(0,0,0,.06); color: #1f2937;
}
.card h3 { margin-top: 0; color: #14532d; }
.card img { width: 100%; border-radius: 12px; }

.result-ok  { border-left: 8px solid #16a34a; }
.result-bad { border-left: 8px solid #dc2626; }
.result-low { border-left: 8px solid #f59e0b; }
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: .8rem; font-weight: 600; color: #fff;
}
.b-ok { background: #16a34a; } .b-bad { background: #dc2626; } .b-low { background: #f59e0b; }
.disease { font-size: 1.7rem; font-weight: 700; margin: 8px 0 2px 0; color: #111827; }
.crop { color: #6b7280; margin-bottom: 14px; }

.bar-wrap { margin: 8px 0; color: #1f2937; font-size: .9rem; }
.bar-bg { background: #e5e7eb; border-radius: 999px; height: 10px; overflow: hidden; }
.bar-fg { background: linear-gradient(90deg, #16a34a, #4ade80); height: 10px; border-radius: 999px; }

.tips { background: #f0fdf4; border-radius: 14px; padding: 14px 18px; margin-top: 14px; color: #14532d; }
.tips li { margin-bottom: 4px; }

.step {
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px;
    padding: 20px; height: 100%; color: #1f2937;
}
.step .n {
    width: 36px; height: 36px; border-radius: 50%; background: #16a34a; color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 700; margin-bottom: 10px;
}
.footer { text-align: center; color: #6b7280; font-size: .85rem; margin-top: 40px; padding: 20px 0; }
.stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Disease advice (general guidance only)
# ----------------------------------------------------------------------------
ADVICE = [
    ("healthy", ["No disease detected. Keep up regular watering and balanced nutrition.",
                 "Inspect leaves weekly so problems are caught early.",
                 "Keep the field free of weeds and fallen debris."]),
    ("late blight", ["Remove and destroy infected leaves and plants.",
                     "Avoid overhead watering and improve air flow between plants.",
                     "Ask your local agriculture officer about a suitable fungicide."]),
    ("early blight", ["Remove lower infected leaves and clear plant debris.",
                      "Rotate crops; avoid planting the same family in the same spot.",
                      "Mulch to stop soil splashing onto leaves."]),
    ("bacterial spot", ["Use disease-free seed and avoid working in wet fields.",
                        "Remove infected plant parts; avoid overhead irrigation.",
                        "Copper-based sprays are commonly advised; check local guidance."]),
    ("powdery mildew", ["Improve air circulation and avoid excess nitrogen.",
                        "Remove badly affected leaves.",
                        "Sulphur-based or other approved sprays may help; check local guidance."]),
    ("rust", ["Remove infected leaves and avoid wetting foliage.",
              "Plant resistant varieties where available.",
              "Approved fungicides may be needed; consult your local expert."]),
    ("scab", ["Rake and destroy fallen infected leaves.",
              "Prune to open up the canopy for better drying.",
              "Preventive fungicide programmes are often used; check local guidance."]),
    ("black rot", ["Prune out infected shoots, fruit and mummified berries.",
                   "Keep the canopy open so leaves dry quickly.",
                   "Follow a locally recommended spray schedule."]),
    ("leaf mold", ["Lower humidity and improve ventilation.",
                   "Avoid wetting leaves when watering.",
                   "Remove infected leaves promptly."]),
    ("septoria", ["Remove infected lower leaves and debris.",
                  "Avoid overhead watering; use drip irrigation.",
                  "Rotate crops for at least a year or two."]),
    ("spider mite", ["Spray leaves with a strong water jet to dislodge mites.",
                     "Encourage natural predators; avoid unnecessary broad-spectrum sprays.",
                     "Neem-based products are a common mild option."]),
    ("target spot", ["Remove infected leaves and improve air flow.",
                     "Avoid leaf wetness late in the day.",
                     "Consult local guidance on fungicides."]),
    ("mosaic virus", ["No cure for virus; remove and destroy infected plants.",
                      "Control insects (aphids, whiteflies) that spread it.",
                      "Wash hands and tools after handling infected plants."]),
    ("yellow leaf curl", ["Control whiteflies, which spread this virus.",
                          "Remove infected plants early; use reflective mulch or nets.",
                          "Plant tolerant varieties where available."]),
    ("leaf scorch", ["Water consistently and avoid drought stress.",
                     "Remove badly affected leaves.",
                     "Improve air circulation around plants."]),
    ("haunglongbing", ["Citrus greening has no cure; report suspected trees to local authorities.",
                       "Control the insect vector (psyllid).",
                       "Remove infected trees as advised locally."]),
    ("citrus greening", ["Citrus greening has no cure; report suspected trees to local authorities.",
                         "Control the insect vector (psyllid).",
                         "Remove infected trees as advised locally."]),
    ("esca", ["Prune out and burn infected wood during dry weather.",
              "Protect pruning wounds.",
              "Consult a local viticulture expert."]),
    ("leaf blight", ["Remove infected leaves and avoid overhead watering.",
                     "Rotate crops and clear crop residue after harvest.",
                     "Use resistant varieties where available."]),
    ("leaf spot", ["Remove infected leaves and avoid wetting foliage.",
                   "Rotate crops and keep the field clean.",
                   "Consult local guidance on fungicides."]),
    ("gray leaf spot", ["Rotate crops and till residue after harvest.",
                        "Plant resistant hybrids where available.",
                        "Consult local guidance on fungicides."]),
]
DEFAULT_ADVICE = ["Isolate the affected plant if possible.",
                  "Remove visibly infected leaves and avoid overhead watering.",
                  "Show the plant to a local agriculture officer or Krishi Vigyan Kendra (KVK)."]


def advice_for(disease: str):
    d = disease.lower()
    for key, tips in ADVICE:
        if key in d:
            return tips
    return DEFAULT_ADVICE


def split_name(name: str):
    crop, _, disease = name.partition("___")
    clean = lambda s: s.replace("_", " ").replace("  ", " ").strip(" ,")
    return clean(crop), clean(disease) or "Unknown"


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model("plant_model.keras")
    with open("class_names.json") as f:
        names = json.load(f)
    return model, names


def predict(img: Image.Image, model):
    arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    return model.predict(arr[None, ...], verbose=0)[0]


def img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>🌿 LeafCare AI</h1>
  <p>Upload a photo of a plant leaf and get an instant disease prediction
     with simple care tips, powered by a deep learning CNN.</p>
  <span class="pill">⚡ Instant results</span>
  <span class="pill">🧠 MobileNetV2 CNN</span>
  <span class="pill">🌾 Built for farmers</span>
</div>
""",
    unsafe_allow_html=True,
)

model_ready = os.path.exists("plant_model.keras") and os.path.exists("class_names.json")
model, class_names = (None, [])
if model_ready:
    with st.spinner("Loading AI model..."):
        model, class_names = load_model()

crops = sorted({split_name(n)[0] for n in class_names})
st.markdown(
    f"""
<div class="stat-row">
  <div class="stat"><b>{len(class_names) or '-'}</b><span>Disease classes</span></div>
  <div class="stat"><b>{len(crops) or '-'}</b><span>Crops supported</span></div>
  <div class="stat"><b>&lt; 2 sec</b><span>Per prediction</span></div>
  <div class="stat"><b>Free</b><span>Runs on your laptop</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------------
tab_detect, tab_how, tab_crops = st.tabs(["🔍 Detect Disease", "📖 How it Works", "🌱 Supported Crops"])

with tab_detect:
    if not model_ready:
        st.error("Model not found. Run `python train.py` first to create "
                 "`plant_model.keras` and `class_names.json`, then refresh this page.")
        st.stop()

    source = st.radio("Choose input", ["Upload a photo", "Use camera"], horizontal=True)
    file = (st.file_uploader("Leaf image (JPG or PNG)", type=["jpg", "jpeg", "png"])
            if source == "Upload a photo" else st.camera_input("Take a photo of the leaf"))

    if file is None:
        st.info("👆 Add a clear, close-up photo of a single leaf to begin. "
                "A plain background and good light give the best results.")
    else:
        img = Image.open(file)
        with st.spinner("Analysing leaf..."):
            probs = predict(img, model)

        top = probs.argsort()[::-1][:3]
        crop, disease = split_name(class_names[top[0]])
        conf = float(probs[top[0]])
        healthy = disease.lower() == "healthy"

        if conf < 0.6:
            cls, badge, label = "result-low", "b-low", "LOW CONFIDENCE"
        elif healthy:
            cls, badge, label = "result-ok", "b-ok", "HEALTHY"
        else:
            cls, badge, label = "result-bad", "b-bad", "DISEASE DETECTED"

        left, right = st.columns([1, 1.15], gap="large")
        with left:
            st.markdown(
                f'<div class="card"><h3>Your leaf</h3>'
                f'<img src="data:image/jpeg;base64,{img_to_b64(img)}"/></div>',
                unsafe_allow_html=True,
            )
        with right:
            bars = ""
            for i in top:
                c, d = split_name(class_names[i])
                p = float(probs[i]) * 100
                bars += (f'<div class="bar-wrap"><div>{c} - {d} <b>({p:.1f}%)</b></div>'
                         f'<div class="bar-bg"><div class="bar-fg" style="width:{p:.1f}%"></div></div></div>')

            tips = "".join(f"<li>{t}</li>" for t in advice_for(disease))
            low_note = ("<p style='color:#b45309'>The model is unsure. Try a closer, sharper "
                        "photo of one leaf in good light.</p>") if conf < 0.6 else ""
            st.markdown(
                f"""
<div class="card {cls}">
  <span class="badge {badge}">{label}</span>
  <div class="disease">{disease}</div>
  <div class="crop">Crop: {crop} &nbsp;|&nbsp; Confidence: <b>{conf*100:.1f}%</b></div>
  {low_note}
  <h4 style="margin-bottom:4px;color:#14532d">Top predictions</h4>
  {bars}
  <div class="tips"><b>💡 General care tips</b><ul style="margin:6px 0 0 18px">{tips}</ul></div>
  <p style="color:#6b7280;font-size:.78rem;margin-top:12px">
    This is an AI-based estimate, not a professional diagnosis. For serious
    outbreaks consult your local agriculture officer or KVK.</p>
</div>
""",
                unsafe_allow_html=True,
            )

with tab_how:
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "Upload", "Add a photo of a single leaf, from your gallery or camera."),
        ("2", "Preprocess", "The image is resized to 224 x 224 pixels for the model."),
        ("3", "CNN Analysis", "MobileNetV2 (transfer learning) extracts patterns like spots and colour changes."),
        ("4", "Result", "You get the disease name, confidence score and care tips."),
    ]
    for col, (n, t, d) in zip((c1, c2, c3, c4), steps):
        col.markdown(f'<div class="step"><div class="n">{n}</div><b>{t}</b><br>'
                     f'<span style="color:#4b5563;font-size:.9rem">{d}</span></div>',
                     unsafe_allow_html=True)
    st.markdown("### Tips for a good photo")
    st.markdown("- Fill the frame with **one leaf**\n- Use **natural daylight**, avoid heavy shadows\n"
                "- Keep the leaf **in focus** against a plain background\n- Photograph the **affected area** clearly")

with tab_crops:
    st.markdown("### Crops and conditions the model knows")
    if not class_names:
        st.warning("Train the model first to see the supported classes.")
    else:
        grouped = {}
        for n in class_names:
            c, d = split_name(n)
            grouped.setdefault(c, []).append(d)
        cols = st.columns(3)
        for idx, (c, ds) in enumerate(sorted(grouped.items())):
            items = "".join(f"<li>{d}</li>" for d in ds)
            cols[idx % 3].markdown(
                f'<div class="card" style="margin-bottom:14px"><h3>🌱 {c}</h3>'
                f'<ul style="margin:0 0 0 18px">{items}</ul></div>',
                unsafe_allow_html=True,
            )

st.markdown('<div class="footer">Made with ❤️ using TensorFlow and Streamlit &nbsp;|&nbsp; '
            'LeafCare AI - College AI Project</div>', unsafe_allow_html=True)
