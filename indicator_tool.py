import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURATION (Matches Web Tool Defaults) ---
DEFAULT_BRIGHTNESS_THRESHOLD = 165
DEFAULT_DETAIL_THRESHOLD = 60

def robust_analyze(image_file, bright_thresh, detail_thresh):
    # 1. Load Image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Get dimensions
    height, width = img.shape[:2]

    # 2. SMART CROP (Matches Web Tool)
    # Crop 10% from each side to remove frame edges/background noise
    margin_x = int(width * 0.1)
    margin_y = int(height * 0.1)
    
    # Ensure crop is valid
    if margin_x > 0 and margin_y > 0:
        cropped = img[margin_y:height-margin_y, margin_x:width-margin_x]
    else:
        cropped = img

    # 3. Convert to Grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # 4. Calculate Global Brightness (Background Check)
    avg_brightness = np.mean(gray)

    # 5. GRID SEARCH (Matches Web Tool)
    # Split into 5x5 grid and calculate variance for each block
    rows, cols = 5, 5
    h, w = gray.shape
    dy, dx = h // rows, w // cols
    
    block_std_devs = []

    for r in range(rows):
        for c in range(cols):
            # Extract block
            y_start = r * dy
            y_end = (r + 1) * dy
            x_start = c * dx
            x_end = (c + 1) * dx
            
            block = gray[y_start:y_end, x_start:x_end]
            
            # Calculate Standard Deviation (Detail) for this block
            if block.size > 0:
                std = np.std(block)
                block_std_devs.append(std)

    # 6. ROBUST SCORING (Top 3 Average)
    # Sort high to low
    block_std_devs.sort(reverse=True)
    
    # Take average of top 3 (or fewer if image is tiny) to filter noise
    top_n = min(3, len(block_std_devs))
    if top_n > 0:
        robust_score = sum(block_std_devs[:top_n]) / top_n
    else:
        robust_score = 0

    # --- LOGIC DECISION TREE ---
    result = {
        "score": robust_score,
        "brightness": avg_brightness,
        "status": "UNKNOWN",
        "reason": "",
        "color": "gray"
    }

    has_text = robust_score >= detail_thresh
    is_clear = avg_brightness >= bright_thresh

    if not has_text:
        # Criteria 1: Blank
        result["status"] = "REJECTED"
        result["reason"] = "Image is Blank (No 'OK' text found)"
        result["color"] = "red"
    elif not is_clear:
        # Criteria 2: OK but Dark Background
        result["status"] = "REJECTED"
        result["reason"] = "'OK' detected, but background is Black/Opaque"
        result["color"] = "red"
    else:
        # Criteria 3: OK + Clear
        result["status"] = "ACCEPTED"
        result["reason"] = "'OK' detected with Clear Background"
        result["color"] = "green"

    return result, cropped

# --- UI SETUP ---
st.set_page_config(page_title="Indicator QC", page_icon="🌡️")

st.title("🌡️ Indicator Analysis Tool")
st.markdown("Upload a cold-chain indicator image to verify its status.")

# Sidebar Settings
st.sidebar.header("Calibration")
b_thresh = st.sidebar.slider("Background Threshold (Brightness)", 0, 255, DEFAULT_BRIGHTNESS_THRESHOLD)
d_thresh = st.sidebar.slider("Text Visibility Threshold (Detail)", 0, 100, DEFAULT_DETAIL_THRESHOLD)
st.sidebar.markdown("""
* **Background:** Lower = More lenient (allows darker images).
* **Detail:** Lower = More sensitive (detects fainter text).
""")

uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Analyze
    res, processed_img = robust_analyze(uploaded_file, b_thresh, d_thresh)
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(uploaded_file, caption="Original Upload", use_container_width=True)
    
    with col2:
        # Display Result
        if res["color"] == "green":
            st.success(f"### {res['status']}")
            st.balloons()
        else:
            st.error(f"### {res['status']}")
        
        st.write(f"**Reason:** {res['reason']}")
        
        # Metrics
        st.markdown("---")
        st.caption("Diagnostics")
        st.metric("Robust Detail Score", f"{res['score']:.1f}", delta_color="normal")
        st.metric("Background Brightness", f"{res['brightness']:.1f}", delta_color="normal")

    # Debug View (Optional)
    with st.expander("See Processed View (Smart Crop)"):
        st.image(processed_img, caption="Analyzed Region (Center 80%)", width=300, channels="BGR")