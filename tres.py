import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Function to calculate and plot histograms ---
def plot_histograms(bgr_image, gray_image):
    """Calculates and plots RGB and Grayscale histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- RGB Histogram ---
    color = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')
    for i, col in enumerate(color):
        hist_rgb = cv2.calcHist([bgr_image], [i], None, [256], [0, 256])
        ax1.plot(hist_rgb, color=col, label=channel_names[i])
    ax1.set_title('Histogram RGB')
    ax1.set_xlabel('Intensity')
    ax1.set_ylabel('Pixel Count')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 256])

    # --- Grayscale Histogram ---
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax2.plot(hist_gray, color='gray')
    ax2.set_title('Histogram Grayscale')
    ax2.set_xlabel('Intensity')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0, 256])
    
    # Return the figure object to display in Streamlit
    return fig

# --- Main App Configuration ---
st.set_page_config(layout="wide", page_title="Image Processing App")

st.title("🖼️ Tugas Pengolahan Citra: Thresholding & Equalization")
st.write("Unggah sebuah gambar untuk melihat hasil pengolahan citra secara otomatis.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read the uploaded image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1) # 1 = Read in color
    
    # Convert BGR (OpenCV default) to RGB for correct display
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Convert to Grayscale for processing
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    st.header("1. Citra Asli dan Grayscale")
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_image, caption='Citra Asli (RGB)', use_column_width=True)
    with col2:
        st.image(gray_image, caption='Citra Grayscale', use_column_width=True)
    
    st.header("2. Analisis Histogram")
    st.pyplot(plot_histograms(original_image, gray_image))

    with st.expander("Lihat Angka Histogram (Data Numerik)"):
        st.write("Data mentah dari histogram. Ditampilkan 25 nilai pertama.")
        hist_gray_data = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        st.text("Grayscale:")
        st.dataframe(hist_gray_data.flatten()[:25])
        
        hist_red_data = cv2.calcHist([original_image], [2], None, [256], [0, 256])
        st.text("Red Channel:")
        st.dataframe(hist_red_data.flatten()[:25])


    st.header("3. Thresholding & Equalization")
    
    # --- Task: Thresholding ---
    threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # --- Task: Equalization ---
    equalized_image = cv2.equalizeHist(gray_image)

    # Display results in columns
    col3, col4 = st.columns(2)
    with col3:
        st.image(binary_image, caption=f'Hasil Threshold (Citra Biner)', use_column_width=True)
        st.success(f"**Nilai Threshold** yang digunakan (dari Otsu's Method): **{int(threshold_value)}**")
        
    with col4:
        st.image(equalized_image, caption='Hasil Histogram Equalization', use_column_width=True)

    # --- Display Equalized Histogram ---
    st.subheader("Perbandingan Histogram Grayscale dan Equalized")
    
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original Grayscale Histogram
    hist_gray_orig = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax3.plot(hist_gray_orig, color='gray')
    ax3.set_title('Histogram Grayscale Asli')
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Pixel Count')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_xlim([0, 256])
    
    # Equalized Histogram
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    ax4.plot(hist_equalized, color='blue')
    ax4.set_title('Histogram Hasil Equalization (Uniform)')
    ax4.set_xlabel('Intensity')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_xlim([0, 256])

    st.pyplot(fig2)

else:
    st.info("☝️ Silakan unggah gambar untuk memulai proses.")