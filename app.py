import streamlit as st
from PIL import Image
import sys
from pathlib import Path
import io

sys.path.insert(0, str(Path(__file__).parent))

from hybrid_ocr import HybridOCR, CorrectionStrategy, format_confidence

st.set_page_config(
    page_title="Chakshu OCR - Local Text Recognition",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ Chakshu OCR - Hybrid OCR System")
st.markdown("""
**100% Local OCR** with intelligent text correction. No cloud services, no GPU required.
Upload an image to extract text with automatic correction.
""")

@st.cache_resource
def get_ocr_engine(strategy):
    return HybridOCR(correction_strategy=strategy, preprocess=True)

with st.sidebar:
    st.header("Settings")
    
    strategy = st.selectbox(
        "Correction Strategy",
        options=["rule_based", "hybrid"],
        index=0,
        help="Rule-based is fast and instant. Hybrid uses LLM for better quality (requires additional packages)."
    )
    
    language = st.selectbox(
        "Language",
        options=["eng", "spa", "fra", "deu"],
        index=0,
        help="Tesseract language pack"
    )
    
    preprocess = st.checkbox("Enable Preprocessing", value=True, help="Enhance image quality before OCR")
    
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
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload an image containing text"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Extract Text", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                try:
                    ocr = get_ocr_engine(strategy)
                    
                    result = ocr.process(
                        image,
                        output_format='detailed',
                        confidence_threshold=confidence_threshold
                    )
                    
                    st.session_state['result'] = result
                    st.success("Text extracted successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    if "Tesseract" in str(e):
                        st.info("Make sure Tesseract OCR is installed on your system.")

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
                help="Text after intelligent correction"
            )
            if st.button("Copy Corrected Text"):
                st.code(result.corrected_text, language=None)
        
        with tab2:
            st.text_area(
                "Raw OCR Output",
                value=result.raw_text,
                height=300,
                help="Direct output from Tesseract"
            )
        
        with tab3:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Average Confidence", f"{result.confidence_avg:.1f}%")
                st.metric("Minimum Confidence", f"{result.confidence_min:.1f}%")
            
            with col_b:
                st.metric("Processing Time", f"{result.processing_time:.2f}s")
                st.metric("Words Detected", result.metadata.get('words_count', 0))
            
            st.markdown("---")
            st.markdown("**Confidence Rating:**")
            st.markdown(format_confidence(result.confidence_avg))
            
            if result.metadata.get('low_confidence_words', 0) > 0:
                st.warning(
                    f"⚠️ {result.metadata['low_confidence_words']} words have low confidence "
                    "and may need manual review."
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
st.markdown(
    "<div style='text-align: center; color: #666;'>Chakshu OCR v0.1.0 - MIT License</div>",
    unsafe_allow_html=True
)
