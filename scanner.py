import cv2
import numpy as np
import streamlit as st
import os
import glob
import io
from PIL import Image
import tempfile
import base64
from pathlib import Path

# Constants
OUTPUT_DIR = Path("dataset/output_images")

# -------------------------------
# PROFESSIONAL CSS BACKGROUND STYLES
# -------------------------------

custom_css = """
<style>
    /* Professional Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Main Container Styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.1rem;
    }
    
    /* Card Styling */
    .css-1r6slb0, .css-1v3fvcr {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    .css-1d391kg .sidebar-content {
        color: white;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Download Button Specific */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    
    /* Image Container Styling */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 5px 20px rgba(0,0,0,0.2);
    }
    
    /* Success/Info/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stError {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: rgba(255,255,255,0.1);
        border-radius: 50px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Spinner Styling */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Metric Cards */
    .stMetric {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255,255,255,0.7);
        margin-top: 2rem;
    }
    
    /* Badge Styling */
    .badge {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #1a1a2e;
        padding: 0.25rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.2);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stImage {
        animation: fadeIn 0.5s ease-out;
    }
</style>
"""

# Apply custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# -------------------------------
# 1. Core image processing functions
# -------------------------------

def order_points(pts):
    """Order contour points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32")
    
    # Sort by x-coordinate and split into left and right
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_most = x_sorted[:2]
    right_most = x_sorted[2:]
    
    # Top-left and bottom-left from left-most points
    left_most = left_most[np.argsort(left_most[:, 1])]
    tl, bl = left_most[0], left_most[1]
    
    # Top-right and bottom-right from right-most points
    right_most = right_most[np.argsort(right_most[:, 1])]
    tr, br = right_most[0], right_most[1]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    """Apply perspective transform to get a top-down view of document."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
    
    return warped

def detect_document_contour(image):
    """Detect document contour and return 4 corner points."""
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Resize for faster processing
    height, width = gray.shape
    target_height = 500
    ratio = height / target_height
    if ratio > 1:
        resized = cv2.resize(gray, (int(width / ratio), target_height))
    else:
        resized = gray.copy()
        ratio = 1
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    
    # Edge detection
    edged = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    image_area = resized.shape[0] * resized.shape[1]
    min_area = image_area * 0.05  # At least 5% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            # Scale back to original size
            return approx.reshape(4, 2) * ratio
    
    return None

def crop_document_only(image):
    """Crop only the document area, removing all background."""
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold to find document content
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find largest contour (document)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Add small margin
    margin = 10
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    # Crop to document
    cropped = image[y:y+h, x:x+w]
    
    return cropped

# ============================================================
# SIRF YEH FUNCTION CHANGE KIYA HAI - BACKGROUND REMOVE KARNE KE LIYE
# ============================================================
def remove_black_background(image):
    """Remove black/dark background and replace with white."""
    if image is None:
        return None
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Better threshold to remove dark/black background
    # Document ko preserve karo, background hatao
    _, mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Dilate mask to include edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    # Create white background
    if len(image.shape) == 3:
        white_bg = np.full_like(image, (255, 255, 255), dtype=np.uint8)
    else:
        white_bg = np.full_like(image, 255, dtype=np.uint8)
    
    # Copy original pixels where mask is non-zero
    result = white_bg.copy()
    result[mask == 255] = image[mask == 255]
    
    return result
# ============================================================
# BAQI CODE BILKUL WAISA HI HAI - KOI CHANGE NAHI
# ============================================================

def enhance_scanned_document(image, bw_mode=False):
    """Enhance the scanned document for better readability."""
    if image is None:
        return None
    
    # First remove any black background
    image = remove_black_background(image)
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Sharpen the image
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    if bw_mode:
        # Convert to pure black and white using Otsu's threshold
        _, bw = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    else:
        return sharpened

def straighten_document(image):
    """Detect and straighten document using Hough Line Transform."""
    if image is None:
        return image
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is not None:
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Consider lines close to horizontal (within 10 degrees)
            if abs(angle) < 10 or abs(angle - 180) < 10:
                angles.append(angle if angle < 90 else angle - 180)
        
        if angles:
            # Calculate median angle
            median_angle = np.median(angles)
            
            # Rotate if angle is significant
            if abs(median_angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=(255, 255, 255))
                return rotated
    
    return image

def process_image(image, bw_mode=False):
    """
    Complete document scanning pipeline:
    1. Detect document contour
    2. Apply perspective correction (straighten)
    3. Crop only document area
    4. Remove black background
    5. Enhance for readability
    """
    original = image.copy()
    
    # Step 1: Detect document contour
    corners = detect_document_contour(original)
    
    if corners is not None:
        # Step 2: Apply perspective correction to straighten the document
        try:
            scanned = four_point_transform(original, corners)
        except Exception as e:
            print(f"Perspective transform failed: {e}")
            scanned = original.copy()
    else:
        # If no document detected, try to straighten the whole image
        scanned = straighten_document(original)
    
    # Step 3: Crop only the document area (remove extra background)
    scanned = crop_document_only(scanned)
    
    # Step 4: Remove black background and replace with white
    scanned = remove_black_background(scanned)
    
    # Step 5: Enhance for better readability
    scanned = enhance_scanned_document(scanned, bw_mode)
    
    return scanned, corners

def save_scanned(image, filename, output_dir="dataset/output_images"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, image)
    return out_path

def get_image_paths(input_dir="dataset/input_images"):
    """Get list of image file paths from input directory."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    return [Path(f) for f in files]

def load_image(image_path):
    """Load an image from file path."""
    return cv2.imread(str(image_path))

def draw_contour(image, contour):
    """Draw the detected contour on the image."""
    if contour is None:
        return image
    img_copy = image.copy()
    cv2.drawContours(img_copy, [contour.astype(int)], -1, (0, 255, 0), 3)
    return img_copy

def batch_process(input_dir="dataset/input_images", output_dir="dataset/output_images", bw_mode=False):
    """Process all images in input_dir and save results."""
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    if not files:
        return 0
    count = 0
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        scanned, _ = process_image(img, bw_mode)
        if scanned is not None:
            base = os.path.basename(f)
            name, _ = os.path.splitext(base)
            out_name = f"{name}_scanned.png"
            save_scanned(scanned, out_name, output_dir)
            count += 1
    return count

def get_image_download_link(img_array, filename, text):
    """Generate a download link for an OpenCV image."""
    ret, png = cv2.imencode('.png', img_array)
    if ret:
        b64 = base64.b64encode(png).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
        return href
    return ""

def convert_to_pdf(image_array, output_pdf_path):
    """Convert a single image to PDF using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    pil_img.save(output_pdf_path, "PDF", resolution=100.0)
    return output_pdf_path

# -------------------------------
# 2. Streamlit UI with Enhanced Header
# -------------------------------

def main():
    st.set_page_config(
        page_title="Document Scanner Pro", 
        layout="wide",
        page_icon="📄",
        initial_sidebar_state="expanded"
    )
    
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 Document Scanner Pro</h1>
        <p>Upload a photo of a document – the system will automatically <strong>straighten, crop, and enhance</strong> it.</p>
        <div class="badge">✨ AI-Powered Document Scanning</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards Row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="feature-card">
            🎯<br>
            <strong>Auto Detect</strong><br>
            <small>Document edges</small>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            🔄<br>
            <strong>Straighten</strong><br>
            <small>Perspective correction</small>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            ✂️<br>
            <strong>Crop Only</strong><br>
            <small>Document area</small>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="feature-card">
            📥<br>
            <strong>Multiple Export</strong><br>
            <small>PNG, JPEG, PDF</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Sidebar settings
    st.sidebar.header("⚙️ Settings")
    bw_mode = st.sidebar.checkbox("Black & White Mode", value=False, 
                                   help="Convert to pure black and white for better text clarity")
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Features:**
    - ✅ Auto-detect document edges
    - ✅ Perspective correction (straighten)
    - ✅ Remove black background → White
    - ✅ Crop only document area
    - ✅ Enhance text clarity
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.success("💡 **Pro Tip:** Place document on a contrasting background for best results!")

    # Main tabs
    tab1, tab2 = st.tabs(["📸 Single Upload", "📦 Batch Process"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Display detection overlay
            corners = detect_document_contour(original)
            if corners is not None:
                overlay = draw_contour(original, corners)
            else:
                overlay = original.copy()
            
            with st.spinner("🔄 Straightening and cropping document..."):
                scanned, detected_corners = process_image(original, bw_mode)
            
            if scanned is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                            caption="📸 Original with Detection", use_container_width=True)
                with col2:
                    st.image(scanned, caption="✨ Scanned & Straightened", 
                            use_container_width=True, clamp=True)
                
                # Download buttons
                st.markdown("### 📥 Download")
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    ret, png = cv2.imencode('.png', scanned)
                    if ret:
                        st.download_button(
                            label="⬇️ PNG",
                            data=png.tobytes(),
                            file_name="scanned_document.png",
                            mime="image/png",
                            use_container_width=True
                        )
                
                with col_dl2:
                    ret, jpg = cv2.imencode('.jpg', scanned, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if ret:
                        st.download_button(
                            label="🖼️ JPEG",
                            data=jpg.tobytes(),
                            file_name="scanned_document.jpg",
                            mime="image/jpeg",
                            use_container_width=True
                        )
                
                with col_dl3:
                    if st.button("📄 Generate PDF", use_container_width=True):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            pdf_path = tmp.name
                        convert_to_pdf(scanned, pdf_path)
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇️ Download PDF", f, 
                                             file_name="scanned.pdf", 
                                             mime="application/pdf",
                                             use_container_width=True)
                        os.unlink(pdf_path)
                
                # Save to folder option
                if st.button("💾 Save to output folder", use_container_width=True):
                    out_path = save_scanned(scanned, "upload_scanned.png")
                    st.success(f"✅ Saved to `{out_path}`")
            else:
                st.error("❌ Could not process image. Please try another image with clear document edges.")

    with tab2:
        st.info("📁 Place images in `dataset/input_images` folder and click the button below.")
        
        # Show existing images
        input_dir = Path("dataset/input_images")
        if input_dir.exists():
            images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
            if images:
                st.write(f"Found **{len(images)}** images ready for batch processing")
        
        if st.button("🚀 Start Batch Processing", use_container_width=True):
            with st.spinner("Processing images..."):
                count = batch_process(bw_mode=bw_mode)
            if count > 0:
                st.success(f"✅ Processed {count} images. Results saved in `dataset/output_images`.")
                
                # Show output files
                output_dir = Path("dataset/output_images")
                if output_dir.exists():
                    st.write("### 📁 Output Files:")
                    for out_file in output_dir.glob("*.png"):
                        st.write(f"- `{out_file.name}`")
            else:
                st.warning("No images found in `dataset/input_images`. Please add some images first.")

    # Professional Footer
    st.markdown("""
    <div class="footer">
        <p>✅ Document Scanner | Straighten | Crop | Remove Black Background | Enhance Text</p>
        <p style="font-size: 0.8rem;">© 2024 Document Scanner Pro | Powered by OpenCV & Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()