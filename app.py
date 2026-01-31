import streamlit as st
import numpy as np
import pickle
from skimage.transform import resize
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoSort Pro | Smart Waste Analytics",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS (Real Estate Theme + Ghost Mode) ---
st.markdown("""
<style>
    /* --- HIDE STREAMLIT BRANDING (Ghost Mode) --- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 1. MAIN BACKGROUND - Dark Luxury Theme */
    .stApp {
        background: linear-gradient(135deg, #1a1c20 0%, #0f1012 100%);
        color: #ffffff;
    }

    /* 2. NAVBAR STYLE */
    .nav-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    /* 3. CARD DESIGN (Like a Property Listing) */
    .property-card {
        background: rgba(30, 30, 30, 0.8);
        border-radius: 20px;
        padding: 0px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid #333;
        overflow: hidden;
        transition: transform 0.3s ease;
        margin-top: 20px;
    }
    .property-card:hover {
        transform: translateY(-5px);
        border-color: #00E676;
    }
    
    /* Card Header (Image Area) */
    .card-header {
        background: #252525;
        padding: 20px;
        text-align: center;
        border-bottom: 1px solid #333;
    }

    /* Card Body (Details) */
    .card-body {
        padding: 25px;
    }

    /* 4. METRICS & BADGES */
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 15px;
    }
    
    .metric-row {
        display: flex;
        justify-content: space-between;
        margin-top: 25px;
        padding-top: 20px;
        border-top: 1px solid #444;
    }
    .metric-item {
        text-align: center;
    }
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 16px;
        font-weight: bold;
        color: #fff;
        margin-top: 5px;
    }

    /* 5. UPLOAD AREA */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.05);
        border: 1px dashed #666;
        border-radius: 15px;
        padding: 30px;
    }
    
    /* 6. BUTTON STYLING */
    div.stButton > button:first-child {
        background-color: #00E676;
        color: #000;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        letter-spacing: 0.5px;
        transition: all 0.3s;
    }
    div.stButton > button:hover {
        background-color: #00C853;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION & MODEL LOADING ---
CATEGORIES = ['Hazardous', 'Non-Recyclable', 'Organic', 'Recyclable']
IMG_SIZE = 64

try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ SYSTEM ERROR: Neural Network Model ('model.pkl') not found. Please re-run training protocol.")
    st.stop()

# --- HERO SECTION (Navbar) ---
st.markdown("""
<div class="nav-container">
    <h1 style="margin:0; font-family: 'Helvetica Neue', sans-serif; font-size: 2.5rem;">
        ♻️ EcoSort <span style="color:#00E676;">Prime</span>
    </h1>
    <p style="color: #888; margin-top:5px; font-size: 1.1rem;">AI-Powered Waste Segregation Dashboard</p>
</div>
""", unsafe_allow_html=True)

# --- MAIN LAYOUT (Two Columns) ---
col1, col2 = st.columns([1, 1.5], gap="large")

# --- LEFT COLUMN: INPUT ---
with col1:
    st.markdown("### 1. Data Source")
    st.write("Upload waste object image for real-time analysis.")
    
    uploaded_file = st.file_uploader("Drop image file here...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Display Preview in a nice container
        st.markdown('<div style="border-radius: 15px; overflow: hidden; margin-top: 20px; border: 1px solid #444;">', unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- RIGHT COLUMN: DASHBOARD ---
with col2:
    st.markdown("### 2. Analytics Engine")
    
    if uploaded_file is None:
        # Placeholder State
        st.info("System Standby. Awaiting Image Input...")
    
    else:
        if st.button("🔍 Run Classification Model", type="primary", use_container_width=True):
            with st.spinner("Processing neural networks..."):
                try:
                    # --- PREPROCESSING (Must match Training Logic) ---
                    img_array = np.array(image.convert('RGB'))
                    img_resized = resize(img_array, (IMG_SIZE, IMG_SIZE, 3))
                    flat_data = img_resized.flatten().reshape(1, -1)
                    
                    # --- PREDICTION ---
                    prediction_index = model.predict(flat_data)[0]
                    result = CATEGORIES[prediction_index]
                    
                    # Confidence Score
                    probs = model.predict_proba(flat_data)
                    confidence = np.max(probs) * 100
                    
                    # --- DYNAMIC STYLING BASED ON RESULT ---
                    if result == "Organic":
                        color = "#00E676" # Green
                        bg_badge = "rgba(0, 230, 118, 0.15)"
                        icon = "🌱"
                        bin_type = "Green Bin (Compost)"
                        action = "Biodegradable Processing"
                    elif result == "Recyclable":
                        color = "#2979FF" # Blue
                        bg_badge = "rgba(41, 121, 255, 0.15)"
                        icon = "♻️"
                        bin_type = "Blue Bin (Recycle)"
                        action = "Material Recovery Facility"
                    elif result == "Hazardous":
                        color = "#FF1744" # Red
                        bg_badge = "rgba(255, 23, 68, 0.15)"
                        icon = "☣️"
                        bin_type = "Hazardous / E-Waste"
                        action = "Specialized Handling Required"
                    else: # Non-Recyclable
                        color = "#B0BEC5" # Grey
                        bg_badge = "rgba(176, 190, 197, 0.15)"
                        icon = "🗑️"
                        bin_type = "Black Bin (Landfill)"
                        action = "General Waste Disposal"

                    # --- THE "PROPERTY CARD" RESULT ---
                    # NOTICE: The HTML string below is shifted all the way to the left. 
                    # DO NOT ADD SPACES IN FRONT OF IT.
                    html_code = f"""
<div class="property-card">
<div class="card-header">
<h2 style="color: {color}; margin:0; font-size: 2rem;">{icon} {result}</h2>
</div>
<div class="card-body">
<div style="display:flex; justify-content:center;">
<span class="status-badge" style="background: {bg_badge}; color: {color}; border: 1px solid {color};">
● AI Confidence: {confidence:.2f}%
</span>
</div>
<p style="text-align:center; color: #ccc; margin-top: 15px; line-height: 1.6;">
System has identified <b>{result}</b> material pattern. 
Recommended disposal protocol initiated.
</p>
<div class="metric-row">
<div class="metric-item">
<div class="metric-label">Target Bin</div>
<div class="metric-value" style="color:{color}">{bin_type}</div>
</div>
<div class="metric-item">
<div class="metric-label">Action</div>
<div class="metric-value">{action}</div>
</div>
<div class="metric-item">
<div class="metric-label">Impact</div>
<div class="metric-value">{'Positive' if result in ['Recyclable', 'Organic'] else 'Neutral'}</div>
</div>
</div>
</div>
</div>
"""
                    st.markdown(html_code, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")