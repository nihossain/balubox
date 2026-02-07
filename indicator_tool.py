import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import shutil
import os

# --- CONFIGURATION ---
# Brightness Threshold:
# Since we now use segmentation to isolate the background, we can be more precise.
# 0 = Black, 255 = White. 
DEFAULT_BG_THRESHOLD = 140 

def configure_tesseract():
    """
    Tries to locate Tesseract in PATH or common Homebrew locations.
    Configures pytesseract to use the found path.
    """
    # 1. Check if 'tesseract' command works directly (is in PATH)
    if shutil.which('tesseract'):
        return True
        
    # 2. Check specific common paths on Mac
    # Homebrew often puts it here on M1/M2/M3 Macs, but it's not always in the system PATH
    possible_paths = [
        '/opt/homebrew/bin/tesseract', # Apple Silicon Macs
        '/usr/local/bin/tesseract',    # Intel Macs
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Tell the library exactly where it is
            pytesseract.pytesseract.tesseract_cmd = path
            return True
            
    return False

def analyze_with_ocr_and_segmentation(image_file, bg_threshold):
    # 1. Load Image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. SMART CROP (Center 80%)
    # Removes edge artifacts that confuse OCR
    h, w = img.shape[:2]
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    cropped = img[margin_y:h-margin_y, margin_x:w-margin_x]
    
    # 3. PRE-PROCESSING
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise before segmentation
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. SEGMENTATION (Otsu's Binarization)
    # Automatically finds the best threshold to separate foreground (ink) from background
    # ret is the calculated threshold, mask is the binary image
    ret, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Determine which part is background. 
    # Assumption: The background occupies the majority of pixels.
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    
    if white_pixels > (total_pixels / 2):
        # Background is white (light), Text is black (dark)
        bg_mask = mask
        text_mask = cv2.bitwise_not(mask)
    else:
        # Background is black (dark), Text is white (light)
        # This happens in the "Black/Opaque" rejection case often
        bg_mask = cv2.bitwise_not(mask)
        text_mask = mask

    # 5. BACKGROUND ANALYSIS
    # Calculate the average brightness of ONLY the background pixels
    # We use the mask to ignore the text/ink pixels in this calculation.
    bg_mean_val = cv2.mean(gray, mask=bg_mask)[0]

    # 6. OPTICAL CHARACTER RECOGNITION (OCR)
    # We use the thresholded image for cleaner OCR
    # Configuration: --psm 6 assumes a single uniform block of text
    try:
        text_found = pytesseract.image_to_string(mask, config='--psm 6')
    except Exception as e:
        text_found = ""
        st.warning(f"OCR Warning: {e}")
    
    # Clean text (remove whitespace, uppercase)
    clean_text = text_found.strip().upper()
    
    # OCR Heuristics: Look for "OK", "0K", "DK", "CK" (common OCR misreads for OK)
    ok_variants = ["OK", "0K", "CK", "DK", "UK", "LK"]
    is_text_ok = any(variant in clean_text for variant in ok_variants)

    # --- DECISION LOGIC ---
    result = {
        "status": "UNKNOWN",
        "reason": "",
        "color": "gray",
        "ocr_text": clean_text,
        "bg_brightness": bg_mean_val
    }

    # CRITERIA 1: Check if Blank (OCR found nothing resembling "OK")
    if not is_text_ok:
        result["status"] = "REJECTED"
        result["reason"] = f"OCR did not find 'OK'. Found: '{clean_text}' (Likely Blank)"
        result["color"] = "red"
        
    # CRITERIA 2: Check Background Brightness (Segmentation Analysis)
    elif bg_mean_val < bg_threshold:
        result["status"] = "REJECTED"
        result["reason"] = f"OCR found 'OK', but background is dark ({bg_mean_val:.1f} < {bg_threshold})"
        result["color"] = "red"
        
    # CRITERIA 3: Accept
    else:
        result["status"] = "ACCEPTED"
        result["reason"] = "OCR found 'OK' and background is clear."
        result["color"] = "green"

    return result, cropped, mask

# --- UI SETUP ---
st.set_page_config(page_title="Indicator OCR Tool", page_icon="👁️")

st.title("👁️ Indicator Analysis (OCR + Segmentation)")

# Dependency Check - Use the new robust configuration function
if not configure_tesseract():
    st.error("❌ **Tesseract OCR not found!**")
    st.markdown("""
    The tool couldn't find the Tesseract application. 
    
    **To fix this on Mac:**
    1. Open Terminal.
    2. Run: `brew install tesseract`
    3. If you installed it but still see this, restart the terminal/script.
    """)
    st.stop()

st.markdown("Uses **Otsu's Segmentation** to isolate the background and **Tesseract OCR** to read the text.")

# Sidebar
st.sidebar.header("Settings")
bg_thresh = st.sidebar.slider("Background Brightness Threshold", 0, 255, DEFAULT_BG_THRESHOLD)

uploaded_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Analyze
    res, cropped_img, mask_img = analyze_with_ocr_and_segmentation(uploaded_file, bg_thresh)
    
    # Layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(cropped_img, caption="Smart Crop", use_container_width=True)
        # Show the segmentation mask (DEBUG)
        st.image(mask_img, caption="Computer Vision Segmentation Mask", use_container_width=True)
    
    with col2:
        if res["color"] == "green":
            st.success(f"### {res['status']}")
            st.balloons()
        else:
            st.error(f"### {res['status']}")
        
        st.write(f"**Reason:** {res['reason']}")
        
        st.markdown("---")
        st.caption("AI Diagnostics")
        st.metric("OCR Readout", res['ocr_text'] if res['ocr_text'] else "[Nothing]")
        st.metric("Segmented BG Brightness", f"{res['bg_brightness']:.1f}")