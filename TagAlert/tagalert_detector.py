#!/usr/bin/env python3
"""
TagAlert Sensor Display Detector
Detects and classifies Sensitec TagAlert temperature sensor displays
as Accept or Reject based on visual analysis.

Usage: python tagalert_detector.py
Press 'q' to quit
"""

import cv2
import numpy as np
import os

# Try to import pytesseract, but make it optional
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Using fallback detection method.")

# Configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.join(SCRIPT_DIR, "TagAlert")
ACCEPT_DIR = os.path.join(TRAINING_DIR, "ACCEPT")
REJECT_DIR = os.path.join(TRAINING_DIR, "REJECT")

# Colors (BGR format for OpenCV)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def load_templates():
    """Load template images from training directories."""
    templates = {"accept": [], "reject": []}
    
    # Load accept templates
    if os.path.exists(ACCEPT_DIR):
        for filename in os.listdir(ACCEPT_DIR):
            filepath = os.path.join(ACCEPT_DIR, filename)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(filepath)
                if img is not None:
                    templates["accept"].append(img)
    
    # Load reject templates
    if os.path.exists(REJECT_DIR):
        for filename in os.listdir(REJECT_DIR):
            filepath = os.path.join(REJECT_DIR, filename)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(filepath)
                if img is not None:
                    templates["reject"].append(img)
    
    print(f"Loaded {len(templates['accept'])} accept templates, {len(templates['reject'])} reject templates")
    return templates


def preprocess_frame(frame):
    """Preprocess frame for better detection."""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def find_display_region(frame, gray):
    """Find the rectangular sensor display region in the frame."""
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    display_regions = []
    
    for contour in contours:
        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Look for rectangular shapes
        if len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(contour)
            
            # Filter by aspect ratio (TagAlert displays are roughly 2:1 to 3:1)
            if 1.2 < aspect_ratio < 4.0 and area > 2000:
                display_regions.append((x, y, w, h, area))
    
    # Sort by area and return largest regions
    display_regions.sort(key=lambda r: r[4], reverse=True)
    return display_regions[:5]  # Return top 5 candidates


def analyze_region(frame, region):
    """
    Analyze a detected region to determine Accept/Reject.
    
    Returns:
        tuple: (classification, confidence)
        classification: "ACCEPT", "REJECT", or None
        confidence: float 0-1
    """
    x, y, w, h = region[:4]
    roi = frame[y:y+h, x:x+w]
    
    if roi.size == 0:
        return None, 0
    
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Analyze the overall brightness
    mean_brightness = np.mean(gray_roi)
    
    # Split the region into left and right halves
    mid_x = w // 2
    left_half = gray_roi[:, :mid_x]
    right_half = gray_roi[:, mid_x:]
    
    left_brightness = np.mean(left_half)
    right_brightness = np.mean(right_half)
    
    # Check for dark background (indicates temperature excursion)
    # Calculate the ratio of dark pixels in the right half (where "OK" appears)
    _, dark_mask = cv2.threshold(right_half, 80, 255, cv2.THRESH_BINARY_INV)
    dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
    
    # Check for numbers in the left region using edge detection
    edges_left = cv2.Canny(left_half, 50, 150)
    edge_density_left = np.sum(edges_left > 0) / edges_left.size
    
    # OCR detection (if available)
    has_number = False
    ok_on_dark = False
    
    if TESSERACT_AVAILABLE:
        try:
            # Check left half for numbers 1-4
            left_roi = roi[:, :mid_x]
            left_text = pytesseract.image_to_string(left_roi, config='--psm 10 -c tessedit_char_whitelist=1234')
            if any(c in left_text for c in ['1', '2', '3', '4']):
                has_number = True
        except Exception:
            pass
    
    # Classification logic
    confidence = 0.5
    
    # REJECT conditions:
    # 1. Dark background detected (right half has significant dark area)
    if dark_ratio > 0.4:
        ok_on_dark = True
        confidence = min(0.5 + dark_ratio, 0.95)
        return "REJECT", confidence
    
    # 2. Number detected in left region (using edge density as fallback)
    if has_number:
        return "REJECT", 0.85
    
    # 3. Edge density suggests text/numbers in left region + dark right side
    if edge_density_left > 0.15 and dark_ratio > 0.3:
        return "REJECT", 0.75
    
    # ACCEPT condition:
    # Light background throughout, no numbers visible
    if dark_ratio < 0.25 and mean_brightness > 120:
        confidence = min(0.6 + (1 - dark_ratio) * 0.3, 0.95)
        return "ACCEPT", confidence
    
    # Uncertain - need more analysis
    return None, 0


def template_match(frame, templates):
    """
    Use template matching as a fallback/secondary method.
    """
    best_match = None
    best_score = 0
    best_loc = None
    best_size = None
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    for classification, template_list in templates.items():
        for template in template_list:
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            
            # Try multiple scales
            for scale in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
                resized = cv2.resize(gray_template, None, fx=scale, fy=scale)
                
                if resized.shape[0] > gray_frame.shape[0] or resized.shape[1] > gray_frame.shape[1]:
                    continue
                
                result = cv2.matchTemplate(gray_frame, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_score:
                    best_score = max_val
                    best_match = classification.upper()
                    best_loc = max_loc
                    best_size = (resized.shape[1], resized.shape[0])
    
    if best_score > 0.5:
        return best_match, best_score, best_loc, best_size
    return None, 0, None, None


def draw_result(frame, x, y, w, h, classification, confidence):
    """Draw the classification result on the frame."""
    color = GREEN if classification == "ACCEPT" else RED
    
    # Draw bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
    
    # Draw label background
    label = f"{classification} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5), font, font_scale, WHITE, thickness)
    
    return frame


def main():
    """Main function to run the webcam detector."""
    print("TagAlert Sensor Display Detector")
    print("=" * 40)
    print("Present the TagAlert sensor display to the camera.")
    print("Press 'q' to quit.\n")
    
    # Load templates
    templates = load_templates()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Webcam opened successfully. Starting detection...\n")
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Preprocess
            gray, blurred = preprocess_frame(frame)
            
            # Find display regions
            regions = find_display_region(frame, gray)
            
            detection_made = False
            
            # Analyze each candidate region
            for region in regions:
                classification, confidence = analyze_region(frame, region)
                
                if classification and confidence > 0.6:
                    x, y, w, h = region[:4]
                    display_frame = draw_result(display_frame, x, y, w, h, 
                                                classification, confidence)
                    detection_made = True
                    break
            
            # If no detection from region analysis, try template matching
            if not detection_made and templates["accept"] or templates["reject"]:
                match, score, loc, size = template_match(frame, templates)
                if match and score > 0.6:
                    x, y = loc
                    w, h = size
                    display_frame = draw_result(display_frame, x, y, w, h, match, score)
                    detection_made = True
            
            # Draw instruction text
            if not detection_made:
                cv2.putText(display_frame, "Present TagAlert sensor to camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            
            # Show the frame
            cv2.imshow("TagAlert Detector", display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDetector closed.")


if __name__ == "__main__":
    main()
