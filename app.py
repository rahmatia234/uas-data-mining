import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Atur gaya visualisasi grafik agar sesuai dengan tema putih-biru
sns.set_theme(style="whitegrid", palette="Blues_r")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- DESAIN & JUDUL UTAMA WEBSITES ---
st.set_page_config(page_title="Data Intelligence Dashboard", page_icon="✨", layout="wide")

# Injeksi CSS Kustom untuk Tampilan Lebih Modern & Elegan
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Background Utama Aplikasi (Soft Mesh Gradient) */
    .stApp {
        background-color: #f8fafc;
        background-image: radial-gradient(at 40% 20%, hsla(228,100%,74%,0.15) 0px, transparent 50%),
                          radial-gradient(at 80% 0%, hsla(189,100%,56%,0.15) 0px, transparent 50%),
                          radial-gradient(at 0% 50%, hsla(355,100%,93%,0.1) 0px, transparent 50%);
    }
    
    /* Styling Header Utama */
    h1 {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Styling Subheaders */
    h2, h3 {
        color: #1e40af !important;
        font-weight: 700 !important;
    }
    h2 {
        padding-bottom: 8px;
        margin-top: 1.5rem;
    }
    
    /* Card/Container styling untuk Metrics (Glassmorphism) */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #3b82f6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.15);
    }
    
    /* Styling DataFrame */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }

    /* Input styling */
    .stSelectbox > div > div > div, .stNumberInput > div > div > input {
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        transition: all 0.3s ease;
    }
    .stSelectbox > div > div > div:focus-within, .stNumberInput > div > div > input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        background: none;
        -webkit-text-fill-color: #1e3a8a;
    }
    
    /* Button Styling (Modern) */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4);
        color: white;
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-size: 3rem;'>✨ Data Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #64748b;'>Platform Eksplorasi Analitik & Prediksi Berbasis Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# --- MEMBUAT MENU NAVIGASI DI SIDEBAR (SEBELAH KIRI) ---
st.sidebar.markdown("<h2 style='text-align: center; color: #1e3a8a;'>📌 Menu Navigasi</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")
menu_pilihan = st.sidebar.radio(
    "Pilih Modul Analisis:",
    ["1. Klasifikasi Diabetes", "2. Clustering Gerai Kopi"]
)
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Gunakan menu di atas untuk beralih antar fitur analitik yang tersedia.")

# ==============================================================================
# MENU 1: KLASIFIKASI DIABETES
# ==============================================================================
if menu_pilihan == "1. Klasifikasi Diabetes":
    st.markdown("<h2 style='color: #1e3a8a;'>🩺 Prediksi Risiko Diabetes</h2>", unsafe_allow_html=True)
    st.info("Modul ini memanfaatkan algoritma Machine Learning untuk melakukan deteksi dini terhadap risiko diabetes berdasarkan data metrik pasien.")
    
    try:
        # 1. Read Dataset
        df_diabetes = pd.read_csv("diabetes.csv", sep=None, engine='python')
        
        st.subheader("Preview Data Diabetes")
        st.dataframe(df_diabetes.head(), use_container_width=True)
        
        # 2. Split Data (Fitur & Target)
        X = df_diabetes.iloc[:, :-1]
        y = df_diabetes.iloc[:, -1]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Pilih Algoritma
        st.subheader("⚙️ Pengaturan Model")
        model_name = st.selectbox(
            "Pilih Algoritma Klasifikasi:",
            ["K-Nearest Neighbors (KNN)", "Naïve Bayes", "Decision Tree"]
        )
        
        if model_name == "K-Nearest Neighbors (KNN)":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "Naïve Bayes":
            model = GaussianNB()
        else:
            model = DecisionTreeClassifier(random_state=42)
            
        # 4. Train Model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # 5. Tampilkan Evaluasi Metrik
        st.markdown("---")
        st.subheader(f"📈 Hasil Evaluasi Performa Model ({model_name})")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{accuracy_score(y_test, y_pred)*100:.1f}%")
        col2.metric("Precision", f"{precision_score(y_test, y_pred)*100:.1f}%")
        col3.metric("Recall", f"{recall_score(y_test, y_pred)*100:.1f}%")
        col4.metric("F1-Score", f"{f1_score(y_test, y_pred)*100:.1f}%")
        
        # Confusion Matrix Graph
        st.write("**Grafik Confusion Matrix:**")
        fig, ax = plt.subplots(figsize=(4, 2.5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel("Prediksi Model")
        plt.ylabel("Kondisi Nyata")
        st.pyplot(fig)
        
        # 6. Form Simulasi Pasien Baru
        st.markdown("---")
        st.subheader("🔍 Simulasi Tebak Kondisi Pasien Baru")
        
        input_values = []
        cols = st.columns(2)
        for i, col_name in enumerate(X.columns):
            with cols[i % 2]:
                val = st.number_input(f"Nilai {col_name}:", value=float(X[col_name].mean()))
                input_values.append(val)
                
        if st.button("Mulai Prediksi Pasien"):
            input_scaled = scaler.transform([input_values])
            hasil = model.predict(input_scaled)[0]
            
            if hasil == 1:
                st.error("🚨 **Hasil:** Pasien berisiko tinggi **Terindikasi DIABETES**.")
            else:
                st.success("✅ **Hasil:** Pasien terindikasi **SEHAT (Negatif Diabetes)**.")
                
    except FileNotFoundError:
        st.error("❌ File 'diabetes.csv' tidak ditemukan! Pastikan file ada di folder yang sama dengan app.py.")

# ==============================================================================
# MENU 2: CLUSTERING GERAI KOPI
# ==============================================================================
elif menu_pilihan == "2. Clustering Gerai Kopi":
    st.markdown("<h2 style='color: #1e3a8a;'>☕ Analisis Spasial & Clustering Gerai Kopi</h2>", unsafe_allow_html=True)
    st.info("Modul ini mengelompokkan lokasi gerai kopi menggunakan algoritma K-Means untuk mengidentifikasi wilayah strategis dan mendeteksi zona minim persaingan.")
    
    try:
        # 1. Read Dataset (Otomatis deteksi pembatas koma/titik koma)
        df_coffee = pd.read_csv("coffee_shops.csv", sep=None, engine='python', encoding='latin1')
        
        st.subheader("Preview Data Gerai Kopi")
        st.dataframe(df_coffee.head(), use_container_width=True)
        
        # 2. Ambil Kolom Angka/Fitur Saja
        X_cluster = df_coffee.select_dtypes(include=[np.number])
        
        if X_cluster.empty:
            st.error("⚠️ File terbaca, tetapi tidak ditemukan kolom berisi angka untuk dilakukan clustering.")
        else:
            scaler_c = StandardScaler()
            X_scaled = scaler_c.fit_transform(X_cluster)
            
            # 3. K-Means Pemodelan
            st.markdown("---")
            k_val = st.slider("Pilih Jumlah Kelompok / Cluster (k):", min_value=2, max_value=5, value=3)
            
            kmeans = KMeans(n_clusters=k_val, random_state=42)
            df_coffee['Cluster'] = kmeans.fit_predict(X_scaled)
            
            # Deteksi Cluster Paling Sepi (Nilai Rata-rata Terkecil)
            cluster_summary = df_coffee.groupby('Cluster')[X_cluster.columns].mean()
            sepi_id = cluster_summary.mean(axis=1).idxmin()
            
            st.warning(f"⚠️ Berdasarkan data, **Cluster {sepi_id}** terdeteksi sebagai **ZONA SEPI / KURANG POTENSIAL**.")
            
            # 4. Visualisasi Grafik Peta Spasial (Scatter Plot)
            st.subheader("🗺️ Peta Distribusi Kelompok Gerai Kopi")
            fig, ax = plt.subplots(figsize=(6, 3.5))
            
            col_x = X_cluster.columns[0]
            col_y = X_cluster.columns[1] if len(X_cluster.columns) > 1 else X_cluster.columns[0]
            
            sns.scatterplot(
                data=df_coffee, x=col_x, y=col_y, 
                hue='Cluster', palette='Blues_r', style='Cluster', s=120, ax=ax, edgecolor='w'
            )
            plt.title("Pengelompokan Lokasi (Cluster)")
            st.pyplot(fig)
            
            # 5. Form Analisis Lokasi Baru
            st.markdown("---")
            st.subheader("📍 Cek Kelayakan Lokasi Baru")
            
            input_loc = []
            cols_c = st.columns(2)
            for i, col_name in enumerate(X_cluster.columns):
                with cols_c[i % 2]:
                    val = st.number_input(f"Nilai {col_name} Lokasi Baru:", value=float(X_cluster[col_name].mean()))
                    input_loc.append(val)
                    
            if st.button("Cek Potensi Lokasi"):
                loc_scaled = scaler_c.transform([input_loc])
                res_cluster = kmeans.predict(loc_scaled)[0]
                
                if res_cluster == sepi_id:
                    st.error(f"🚨 **Hasil Analisis:** Lokasi ini masuk ke **Cluster {res_cluster} (ZONA SEPI)**. Tidak disarankan membuka usaha di sini.")
                else:
                    st.success(f"✅ **Hasil Analisis:** Lokasi ini masuk ke **Cluster {res_cluster} (ZONA POTENSIAL)**. Layak untuk dipertimbangkan!")
                    
    except FileNotFoundError:
        st.error("❌ File 'coffee_shops.csv' tidak ditemukan! Pastikan file ada di folder yang sama dengan app.py.")