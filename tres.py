import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Fungsi-Fungsi Pembantu ---

def plot_rgb_histogram(bgr_image):
    """Menghitung dan menampilkan plot histogram RGB."""
    fig, ax = plt.subplots(figsize=(8, 5))
    color = ('b', 'g', 'r')
    channel_names = ('Blue', 'Green', 'Red')
    
    for i, col in enumerate(color):
        hist_rgb = cv2.calcHist([bgr_image], [i], None, [256], [0, 256])
        ax.plot(hist_rgb, color=col, label=channel_names[i])
        
    ax.set_title('Histogram RGB')
    ax.set_xlabel('Intensitas')
    ax.set_ylabel('Jumlah Piksel')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim([0, 256])
    return fig

def plot_threshold_analysis(gray_image, binary_image, threshold_value):
    """Menampilkan analisis thresholding: histogram dengan garis threshold dan citra binernya."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Kiri: Histogram Grayscale dengan Garis Threshold
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax1.plot(hist_gray, color='gray')
    ax1.axvline(x=threshold_value, color='r', linestyle='--', label=f'Threshold = {int(threshold_value)}')
    ax1.set_title('Histogram Grayscale dan Garis Threshold')
    ax1.set_xlabel('Intensitas')
    ax1.set_ylabel('Jumlah Piksel')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 256])

    # Kanan: Citra Biner Hasil Thresholding
    ax2.imshow(binary_image, cmap='gray')
    ax2.set_title('Hasil Citra Biner')
    ax2.axis('off')

    return fig

def plot_equalization_analysis(gray_image, equalized_image):
    """Menampilkan perbandingan histogram asli dan hasil equalisasi."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Kiri: Histogram Asli
    hist_gray_orig = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    ax1.plot(hist_gray_orig, color='gray')
    ax1.set_title('Histogram Grayscale Asli')
    ax1.set_xlabel('Intensitas')
    ax1.set_ylabel('Jumlah Piksel')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim([0, 256])

    # Kanan: Histogram Hasil Equalization
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    ax2.plot(hist_equalized, color='blue')
    ax2.set_title('Histogram Hasil Equalization (Uniform)')
    ax2.set_xlabel('Intensitas')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_xlim([0, 256])
    
    return fig

# --- Konfigurasi Halaman Utama Aplikasi ---
st.set_page_config(layout="wide", page_title="Analisis Citra Digital")

st.title("🖼️ Analisis Citra: Thresholding & Equalization")
st.write("Aplikasi ini menjalankan analisis sesuai tugas: menampilkan histogram, melakukan thresholding otomatis (Otsu), dan histogram equalization.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Pilih file gambar...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 1. Baca dan siapkan gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 2. Lakukan semua kalkulasi terlebih dahulu
    threshold_value, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    equalized_image = cv2.equalizeHist(gray_image)

    # --- Tampilan Hasil ---

    st.header("1. Citra Input")
    col1, col2 = st.columns(2)
    with col1:
        st.image(rgb_image, caption='Citra Asli (RGB)', use_column_width=True)
    with col2:
        st.image(gray_image, caption='Citra Grayscale', use_column_width=True)

    st.divider()

    st.header("2. Analisis Histogram")
    st.pyplot(plot_rgb_histogram(original_image))
    with st.expander("Lihat Angka Histogram (Data Numerik)"):
        hist_gray_data = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        st.write("Data mentah dari histogram Grayscale (25 nilai pertama):")
        st.dataframe(hist_gray_data.flatten()[:25].reshape(1, -1))

    st.divider()

    st.header("3. Analisis Thresholding (Metode Otsu)")
    st.info(f"Metode Otsu secara otomatis menganalisis histogram dan menemukan nilai threshold terbaik di: **{int(threshold_value)}**")
    st.pyplot(plot_threshold_analysis(gray_image, binary_image, threshold_value))
    
    st.divider()

    st.header("4. Analisis Histogram Equalization")
    st.image(equalized_image, caption='Citra Hasil Equalization', use_column_width='auto', width=400)
    st.pyplot(plot_equalization_analysis(gray_image, equalized_image))

else:
    st.info("☝️ Silakan unggah gambar untuk memulai proses.")