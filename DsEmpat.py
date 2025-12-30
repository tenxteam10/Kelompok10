import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Set page config
st.set_page_config(page_title="Plate Detection Dashboard", layout="wide")

# Custom CSS to mimic the template style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Ubuntu:wght@300;400;500;700&display=swap');

* {
    font-family: 'Ubuntu', sans-serif;
}

.sidebar .sidebar-content {
    background-color: #2a2185;
    color: white;
}

.main {
    background-color: #f5f5f5;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 7px 25px rgba(0, 0, 0, 0.08);
    margin: 10px;
    color: black;  /* Ensure text is black */
}

.card:hover {
    background: #2a2185;
    color: black;  /* Keep text black on hover */
}

.table {
    width: 100%;
    border-collapse: collapse;
}

.table th, .table td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

.status {
    padding: 2px 4px;
    border-radius: 4px;
    color: white;
    font-weight: 500;
}

.delivered { background: #8de02c; }
.pending { background: #e9b10a; }
.return { background: #f00; }
.inProgress { background: #1795ce; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
menu = ["Home", "Detection Steps", "Full Detection Process", "Settings"]
choice = st.sidebar.selectbox("Menu", menu)

# Initialize session state for results
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []

# Default parameters
default_params = {
    'canny_min': 30,
    'canny_max': 150,
    'kernel_w': 15,
    'kernel_h': 5,
    'min_area': 1000,
    'min_aspect': 2.0,
    'max_aspect': 6.0
}

# Main content
if choice == "Home":
    st.title("Plate Detection Dashboard")
    
    # Topbar simulation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Toggle Sidebar"):
            st.sidebar.empty()  # Simple toggle simulation
    with col2:
        search = st.text_input("Search here", "")
    with col3:
        st.image("https://via.placeholder.com/40", caption="User", width=40)
    
    # Cards
    st.subheader("Statistics")
    total_images = len(st.session_state.detection_results)
    detected_plates = sum(1 for r in st.session_state.detection_results if r['plates_found'] > 0)
    success_rate = (detected_plates / total_images * 100) if total_images > 0 else 0
    efficiency = success_rate * 10  # Dummy calculation
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="card"><div class="numbers">{total_images}</div><div class="cardName">Images Processed</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="numbers">{detected_plates}</div><div class="cardName">Plates Detected</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="numbers">{success_rate:.1f}%</div><div class="cardName">Success Rate</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="card"><div class="numbers">{efficiency:.1f}</div><div class="cardName">Efficiency Score</div></div>', unsafe_allow_html=True)
    
    # Tables
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Recent Detections")
        if st.session_state.detection_results:
            table_data = [
                [r['filename'], f"{r['plates_found']} plates", "Detected" if r['plates_found'] > 0 else "Not Detected", "Completed"]
                for r in st.session_state.detection_results[-5:]  # Last 5
            ]
            st.table(table_data)
        else:
            st.write("No detections yet.")
    
    with col2:
        st.subheader("Recent Images")
        if st.session_state.detection_results:
            for r in st.session_state.detection_results[-2:]:  # Last 2
                st.image(r['original_image'], caption=r['filename'], width=100)
        else:
            st.image("https://via.placeholder.com/100", caption="Sample Image")

elif choice == "Detection Steps":
    st.title("Detection Steps")
    st.write("Upload an image to see the individual steps: Edge Detection, Morphological Transformation, and Contour Filtering. Compare before (default settings) and after (custom settings).")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="steps")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Get custom parameters
        custom_params = {
            'canny_min': st.session_state.get('canny_min', 30),
            'canny_max': st.session_state.get('canny_max', 150),
            'kernel_w': st.session_state.get('kernel_w', 15),
            'kernel_h': st.session_state.get('kernel_h', 5),
            'min_area': st.session_state.get('min_area', 1000),
            'min_aspect': st.session_state.get('min_aspect', 2.0),
            'max_aspect': st.session_state.get('max_aspect', 6.0)
        }
        
        # Function to process image with given params
        def process_steps(img, params):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, params['canny_min'], params['canny_max'])
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (params['kernel_w'], params['kernel_h']))
            morph = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_with_boxes = img.copy()
            cropped_plates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > params['min_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    if params['min_aspect'] < aspect_ratio < params['max_aspect']:
                        cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cropped = img[y:y+h, x:x+w]
                        cropped_plates.append(cropped)
            
            return edged, morph, img_with_boxes, cropped_plates
        
        # Process with default and custom
        edged_default, morph_default, detected_default, crops_default = process_steps(img_cv, default_params)
        edged_custom, morph_custom, detected_custom, crops_custom = process_steps(img_cv, custom_params)
        
        # Display comparison
        st.subheader("Comparison: Default Settings vs. Custom Settings")
        
        # Edge Detection
        st.subheader("1. Edge Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            st.image(cv2.cvtColor(edged_default, cv2.COLOR_GRAY2RGB), caption="Default Edges", use_container_width=True)
        with col2:
            st.write("**Custom**")
            st.image(cv2.cvtColor(edged_custom, cv2.COLOR_GRAY2RGB), caption="Custom Edges", use_container_width=True)
        
        # Morphological Transformation
        st.subheader("2. Morphological Transformation")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            st.image(cv2.cvtColor(morph_default, cv2.COLOR_GRAY2RGB), caption="Default Morphed", use_container_width=True)
        with col2:
            st.write("**Custom**")
            st.image(cv2.cvtColor(morph_custom, cv2.COLOR_GRAY2RGB), caption="Custom Morphed", use_container_width=True)
        
        # Contour Filtering
        st.subheader("3. Contour Filtering & Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Default**")
            detected_default_rgb = cv2.cvtColor(detected_default, cv2.COLOR_BGR2RGB)
            st.image(detected_default_rgb, caption="Default Detection", use_container_width=True)
            if crops_default:
                st.write("Cropped Plates:")
                for i, crop in enumerate(crops_default):
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"Plate {i+1}", width=150)
            else:
                st.write("No plates detected.")
        with col2:
            st.write("**Custom**")
            detected_custom_rgb = cv2.cvtColor(detected_custom, cv2.COLOR_BGR2RGB)
            st.image(detected_custom_rgb, caption="Custom Detection", use_container_width=True)
            if crops_custom:
                st.write("Cropped Plates:")
                for i, crop in enumerate(crops_custom):
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"Plate {i+1}", width=150)
            else:
                st.write("No plates detected.")

elif choice == "Full Detection Process":
    st.title("Full Detection Process")
    st.write("Upload an image to see the complete combination of Edge Detection, Morphological Transformation, and Contour Filtering for plate detection without OCR.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="full")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Step 1: Edge Detection
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        canny_min = st.session_state.get('canny_min', 30)
        canny_max = st.session_state.get('canny_max', 150)
        edged = cv2.Canny(blurred, canny_min, canny_max)
        
        # Step 2: Morphological Transformation
        kernel_w = st.session_state.get('kernel_w', 15)
        kernel_h = st.session_state.get('kernel_h', 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
        morph = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Step 3: Contour Filtering
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_area = st.session_state.get('min_area', 1000)
        min_aspect = st.session_state.get('min_aspect', 2)
        max_aspect = st.session_state.get('max_aspect', 6)
        
        img_with_boxes = img_cv.copy()
        cropped_plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if min_aspect < aspect_ratio < max_aspect:
                    cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cropped = img_cv[y:y+h, x:x+w]
                    cropped_plates.append(cropped)
        
        # Display steps
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Edge Detection")
            st.image(cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB), caption="Edges", use_container_width=True)
        with col2:
            st.subheader("2. Morphological Transformation")
            st.image(cv2.cvtColor(morph, cv2.COLOR_GRAY2RGB), caption="Morphed", use_container_width=True)
        with col3:
            st.subheader("3. Contour Filtering & Detection")
            detected_img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            st.image(detected_img_rgb, caption="Final Detection", use_container_width=True)
        
        if cropped_plates:
            st.subheader("Cropped Plates")
            cols = st.columns(min(len(cropped_plates), 3))
            for i, crop in enumerate(cropped_plates):
                with cols[i % 3]:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"Plate {i+1}", width=200)
        else:
            st.write("No plates detected.")
        
        # Save result to session state
        result = {
            'filename': uploaded_file.name,
            'original_image': image,
            'detected_image': detected_img_rgb,
            'plates_found': len(cropped_plates),
            'cropped_plates': [cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) for crop in cropped_plates]
        }
        st.session_state.detection_results.append(result)

elif choice == "Settings":
    st.title("Settings")
    st.write("Configure detection parameters.")
    
    # Sliders for parameters
    st.session_state.canny_min = st.slider("Canny Min Threshold", 0, 255, st.session_state.get('canny_min', 30))
    st.session_state.canny_max = st.slider("Canny Max Threshold", 0, 255, st.session_state.get('canny_max', 150))
    st.session_state.kernel_w = st.slider("Kernel Width", 1, 50, st.session_state.get('kernel_w', 15))
    st.session_state.kernel_h = st.slider("Kernel Height", 1, 50, st.session_state.get('kernel_h', 5))
    st.session_state.min_area = st.slider("Minimum Contour Area", 100, 10000, st.session_state.get('min_area', 1000))
    st.session_state.min_aspect = st.slider("Minimum Aspect Ratio", 1.0, 10.0, st.session_state.get('min_aspect', 2.0))
    st.session_state.max_aspect = st.slider("Maximum Aspect Ratio", 1.0, 10.0, st.session_state.get('max_aspect', 6.0))
    
    if st.button("Reset to Defaults"):
        st.session_state.canny_min = 30
        st.session_state.canny_max = 150
        st.session_state.kernel_w = 15
        st.session_state.kernel_h = 5
        st.session_state.min_area = 1000
        st.session_state.min_aspect = 2.0
        st.session_state.max_aspect = 6.0
        st.success("Settings reset to defaults.")
