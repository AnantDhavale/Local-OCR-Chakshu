import streamlit as st
from PIL import Image
import sys
from pathlib import Path
import io
import os

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_ocr import HybridOCR, CorrectionStrategy, format_confidence

st.set_page_config(
    page_title="Chakshu OCR - Local Text Recognition",
    page_icon="👁️",
    layout="wide"
)

# Password Protection
def check_password():
    """Returns `True` if the user has entered the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == os.environ.get("APP_PASSWORD", ""):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
        else:
            st.session_state["password_correct"] = False

    # First run, show password input
    if "password_correct" not in st.session_state:
        st.markdown("### 🔒 Authentication Required")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.caption("Enter password to access the OCR system")
        return False
    
    # Password incorrect, show input + error
    elif not st.session_state["password_correct"]:
        st.markdown("### 🔒 Authentication Required")
        st.text_input(
            "Password", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("❌ Incorrect password")
        return False
    
    # Password correct
    else:
        return True

if not check_password():
    st.stop()

st.title("👁️ Chakshu OCR - Hybrid OCR System")
st.markdown("""
**100% Local OCR** with intelligent text correction. No cloud services, no GPU required.
Upload an image to extract text with automatic correction.
""")

@st.cache_resource
def get_ocr_engine(strategy, preset, language):
    """Get cached OCR engine based on parameters to avoid reinitializing"""
    return HybridOCR(correction_strategy=strategy, preprocess=True, preprocess_preset=preset, language=language)

def clear_result():
    """Clear previous OCR result from session state"""
    if 'result' in st.session_state:
        del st.session_state['result']

with st.sidebar:
    st.header("Settings")
    
    strategy = st.selectbox(
        "Correction Strategy",
        options=["rule_based", "hybrid"],
        index=0,
        help="✨ IMPROVED: Now with dictionary-based spell checking and character confusion patterns!"
    )
    
    language = st.selectbox(
        "Language",
        options=["eng", "spa", "fra", "deu"],
        index=0,
        help="Tesseract language pack"
    )
    
    preset = st.selectbox(
        "Preprocessing Preset",
        options=["default", "document", "diagram", "photo", "low_quality", "newspaper", "manuscript", "minimal"],
        index=0,
        help="🎯 Choose preset: document (clean text), diagram (flowcharts/boxes), photo (text in photos), low_quality (poor scans), newspaper (old prints), manuscript (medieval/old texts - no correction), minimal (no preprocessing)"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0,
        max_value=100,
        value=0,
        help="Filter out words below this confidence level"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    - **Privacy First**: All processing happens locally
    - **No Cost**: Free forever, no API charges
    - **Offline**: Works without internet
    - **Fast**: 3-5 seconds per page
    - **Open Source**: MIT Licensed
    """)
    
    st.markdown("---")
    st.markdown("### 🆘 Troubleshooting")
    st.markdown("""
    **Output looks like gibberish?**
    - Try the "minimal" preset
    - Check image resolution (200+ DPI)
    - Try uploading a cleaner/brighter image
    
    **Missing text?**
    - Increase contrast in photo/image editor
    - Use a higher quality scan
    - Try "low_quality" preset if scan is poor
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload an image containing text",
        on_change=clear_result
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file as bytes and convert to PIL Image
            image = Image.open(io.BytesIO(uploaded_file.read()))
            
            # Validate image size (max 10MB)
            file_size_mb = uploaded_file.size / (1024 * 1024)
            if file_size_mb > 10:
                st.error(f"⚠️ Image too large ({file_size_mb:.1f}MB). Maximum is 10MB.")
            else:
                # Show image and metadata
                col_img_info, col_img_meta = st.columns([3, 1])
                with col_img_info:
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                with col_img_meta:
                    st.caption(f"📏 {image.width}x{image.height}px")
                    st.caption(f"📁 {file_size_mb:.1f}MB")
        except Exception as e:
            st.error(f"❌ Failed to load image: {str(e)}")
            image = None
        
        if st.button("Extract Text", type="primary", use_container_width=True):
            if image is None:
                st.error("❌ Please upload a valid image first")
            else:
                with st.spinner("🔄 Processing image... (this may take a moment)"):
                    try:
                        ocr = get_ocr_engine(strategy, preset, language)
                        
                        result = ocr.process(
                            image,
                            output_format='detailed',
                            confidence_threshold=confidence_threshold
                        )
                        
                        st.session_state['result'] = result
                        st.success("✅ Text extracted successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        if "Tesseract" in str(e):
                            st.info("ℹ️ Tesseract OCR may not be installed. Contact support if issue persists.")

with col2:
    st.subheader("Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        tab1, tab2, tab3 = st.tabs(["Corrected Text", "Original Text", "Statistics"])
        
        with tab1:
            st.text_area(
                "Corrected Text",
                value=result.corrected_text,
                height=300,
                help="Text after intelligent correction",
                disabled=True
            )
            col_copy, col_download = st.columns(2)
            with col_copy:
                if st.button("📋 Copy to Clipboard", use_container_width=True):
                    st.info("📋 Text copied! Paste it anywhere with Ctrl+V")
            with col_download:
                # Download corrected text as file
                st.download_button(
                    label="⬇️ Download Text",
                    data=result.corrected_text,
                    file_name="ocr_result.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with tab2:
            st.text_area(
                "Raw OCR Output",
                value=result.raw_text,
                height=300,
                help="Direct output from Tesseract",
                disabled=True
            )
            if st.button("⬇️ Download Raw Text", use_container_width=True):
                st.download_button(
                    label="⬇️ Download Raw Text",
                    data=result.raw_text,
                    file_name="ocr_raw.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with tab3:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("✅ Average Confidence", f"{result.confidence_avg:.1f}%")
                st.metric("⚠️ Minimum Confidence", f"{result.confidence_min:.1f}%")
            
            with col_b:
                st.metric("⏱️ Processing Time", f"{result.processing_time:.2f}s")
                st.metric("📝 Words Detected", result.metadata.get('words_count', 0))
            
            st.markdown("---")
            
            # Display confidence distribution
            st.markdown("**OCR Quality Analysis:**")
            confidence_rating = format_confidence(result.confidence_avg)
            if confidence_rating:
                st.markdown(confidence_rating)
            
            # Detailed quality breakdown
            high_conf = result.metadata.get('high_confidence_words', 0)
            med_conf = result.metadata.get('medium_confidence_words', 0)
            low_conf = result.metadata.get('low_confidence_words', 0)
            
            if high_conf + med_conf + low_conf > 0:
                st.markdown("**Word Confidence Distribution:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High (>70%)", high_conf)
                with col2:
                    st.metric("Medium (40-70%)", med_conf)
                with col3:
                    st.metric("Low (<40%)", low_conf)
            
            if low_conf > 0:
                st.warning(
                    f"⚠️ {low_conf} words have low confidence "
                    "and may need manual review."
                )
            
            # Export results as JSON
            st.markdown("---")
            st.markdown("**Export Results:**")
            json_data = result.to_json(pretty=True)
            st.download_button(
                label="📄 Download Full Results (JSON)",
                data=json_data,
                file_name="ocr_results.json",
                mime="application/json",
                use_container_width=True
            )
    else:
        st.info("Upload an image and click 'Extract Text' to see results here.")

st.markdown("---")

with st.expander("Sample Images & Tips"):
    st.markdown("""
    **For Best Results:**
    - Use high-resolution images (at least 300 DPI)
    - Ensure good contrast between text and background
    - Avoid skewed or rotated images (preprocessing helps)
    - Use clear, printed text (handwriting may not work well)
    
    **Supported Formats:** PNG, JPG, JPEG, TIFF, BMP
    """)

st.markdown("---")

st.markdown("---")

with st.expander("⚠️ Known Limitations & Tips"):
    st.markdown("""
    **Medieval Manuscripts & Old Texts:**
    - Old English/Middle English handwriting is very challenging for modern OCR
    - Try the "minimal" preset (zero preprocessing) for best results
    - If output is still poor, the manuscript font may be incompatible with modern Tesseract
    - **Solution**: Try uploading a cleaner image or higher resolution scan
    
    **For Best Results:**
    - **Clean documents**: Use "document" preset
    - **Diagrams/flowcharts**: Use "diagram" preset  
    - **Medieval texts**: Use "minimal" preset (or "manuscript" with no correction)
    - **Photos with text**: Use "photo" preset
    - **Poor quality scans**: Use "low_quality" preset
    
    **If output is still corrupted:**
    - Try "minimal" preset (no preprocessing whatsoever)
    - Check if image resolution is at least 200 DPI
    - Medieval manuscripts may need specialized OCR software
    """)

with st.expander("🎉 What's New in This Version"):
    st.markdown("""
    **Major Improvements:**
    
    1. **Dictionary-Based Spell Checking** ✨
       - Automatically corrects misspelled words using dictionary
       - Only corrects low-confidence words to avoid false fixes
    
    2. **Character Confusion Patterns** 🔤
       - Fixes common OCR mistakes (rn→m, vv→w, 0→O, 1→l/I)
       - Number/letter confusion detection
    
    3. **5x Faster Deskew** ⚡
       - Improved from 5 OCR calls to just 1
       - Uses single OSD detection for rotation
    
    4. **Adaptive Binarization** 🎨
       - Otsu's method instead of fixed threshold
       - Automatically adapts to image lighting
    
    5. **Preprocessing Presets** 🎯
       - Optimized presets for different document types
       - Document, photo, low-quality, newspaper options
    
    6. **Confidence-Based Filtering** 🎲
       - Only corrects words below confidence threshold
       - Preserves high-confidence text untouched
    """)

st.markdown(
    "<div style='text-align: center; color: #666;'>Chakshu OCR v0.2.0 - MIT License</div>",
    unsafe_allow_html=True
)
