import streamlit as st
import pandas as pd
import numpy as np
import sckit as skl

# Coba import plotly.express, jika gagal, gunakan alternatif atau pesan error
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Library 'plotly' tidak terinstall. Jalankan 'pip install plotly' di terminal Anda untuk mengaktifkan visualisasi interaktif.")

# Coba import sklearn, jika gagal, gunakan alternatif atau pesan error
try:
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Library 'scikit-learn' tidak terinstall. Jalankan 'pip install scikit-learn' di terminal Anda untuk mengaktifkan fitur prediksi.")

def generate_sales_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    products = ['Produk A', 'Produk B', 'Produk C']
    locations = [
        {'name': 'Jakarta', 'lat': -6.2088, 'lon': 106.8456},
        {'name': 'Surabaya', 'lat': -7.2575, 'lon': 112.7521},
        {'name': 'Bandung', 'lat': -6.9175, 'lon': 107.6191}
    ]
    
    data = []
    for date in dates:
        for prod in products:
            loc = np.random.choice(locations)
            sales = np.random.randint(10, 100)
            price = np.random.uniform(50000, 200000)  # Harga dalam Rupiah, ditingkatkan untuk realisme
            revenue = sales * price
            data.append({
                'tanggal': date,
                'produk': prod,
                'lokasi': loc['name'],
                'lat': loc['lat'],
                'lon': loc['lon'],
                'jumlah_penjualan': sales,
                'harga': price,
                'pendapatan': revenue
            })
    return pd.DataFrame(data)

# Inisialisasi session state untuk menyimpan data pengguna - dimulai dengan DataFrame kosong
if 'sales_df' not in st.session_state:
    st.session_state.sales_df = pd.DataFrame(columns=['tanggal', 'produk', 'lokasi', 'lat', 'lon', 'jumlah_penjualan', 'harga', 'pendapatan'])

# Fungsi untuk mendapatkan koordinat berdasarkan lokasi
def get_location_coords(location):
    coords = {
        'Jakarta': {'lat': -6.2088, 'lon': 106.8456},
        'Surabaya': {'lat': -7.2575, 'lon': 112.7521},
        'Bandung': {'lat': -6.9175, 'lon': 107.6191}
    }
    return coords.get(location, {'lat': 0, 'lon': 0})

def train_sales_model(df):
    if not SKLEARN_AVAILABLE:
        return None, None, None
    if df.empty:
        return None, None, None
    # Persiapan data
    encoder = OneHotEncoder(sparse_output=False)
    encoded_features = encoder.fit_transform(df[['produk', 'lokasi']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['produk', 'lokasi']))
    
    # Gabungkan fitur: hari, produk_encoded, lokasi_encoded
    df['hari'] = df['tanggal'].dt.day
    X = pd.concat([df[['hari']], encoded_df], axis=1)
    y = df['jumlah_penjualan']
    
    # Check jika data terlalu sedikit untuk split train/test
    if len(X) <= 1:
        # Jika hanya 1 sampel atau kurang, latih model pada semua data tanpa split
        model = LinearRegression()
        model.fit(X, y)
        mse = None  # Tidak dapat menghitung MSE karena tidak ada test set
        print("Data terlalu sedikit untuk menghitung MSE. Model dilatih pada semua data.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error: {mse}')
    
    return model, encoder, mse

st.sidebar.title("Dashboard Penjualan")
page = st.sidebar.radio("Pilih Halaman", ["Input Data", "Ringkasan", "Visualisasi", "Prediksi"])

if page == "Input Data":
    st.title("Input Data Penjualan")
    
    st.subheader("Tambahkan Data Penjualan Baru")
    with st.form("input_form"):
        tanggal = st.date_input("Tanggal Penjualan")
        produk = st.text_input("Nama Produk")  # Diubah menjadi text_input untuk fleksibilitas, sehingga pengguna dapat mengetik nama produk apa saja
        lokasi = st.selectbox("Lokasi", options=['Jakarta', 'Surabaya', 'Bandung'])
        jumlah_penjualan = st.number_input("Jumlah Penjualan", min_value=0, value=0)
        harga = st.number_input("Harga per Unit (Rp)", min_value=0.0, value=0.0)  # Label diperbarui untuk Rupiah
        
        submitted = st.form_submit_button("Tambahkan Data")
        if submitted:
            if not produk.strip():  # Validasi agar produk tidak kosong
                st.error("Nama produk tidak boleh kosong!")
            else:
                coords = get_location_coords(lokasi)
                pendapatan = jumlah_penjualan * harga  # Hitung pendapatan dalam Rupiah
                new_data = pd.DataFrame({
                    'tanggal': [pd.to_datetime(tanggal)],
                    'produk': [produk.strip()],
                    'lokasi': [lokasi],
                    'lat': [coords['lat']],
                    'lon': [coords['lon']],
                    'jumlah_penjualan': [jumlah_penjualan],
                    'harga': [harga],
                    'pendapatan': [pendapatan]
                })
                st.session_state.sales_df = pd.concat([st.session_state.sales_df, new_data], ignore_index=True)
                st.success("Data berhasil ditambahkan!")
    
    st.subheader("Data Penjualan Saat Ini")
    st.dataframe(st.session_state.sales_df)
    
    # Opsi untuk reset data ke simulasi awal
    if st.button("Reset ke Data Simulasi"):
        st.session_state.sales_df = generate_sales_data()
        st.success("Data direset ke simulasi awal!")

elif page == "Ringkasan":
    st.title("Ringkasan Penjualan")
    
    sales_df = st.session_state.sales_df
    
    # Filter tanggal untuk interaktivitas (opsional, agar ringkasan berubah berdasarkan input)
    start_date = st.date_input("Pilih Tanggal Mulai", value=pd.to_datetime('2023-01-01'))
    end_date = st.date_input("Pilih Tanggal Akhir", value=pd.to_datetime('2023-12-31'))
    filtered_df = sales_df[(sales_df['tanggal'] >= pd.to_datetime(start_date)) & (sales_df['tanggal'] <= pd.to_datetime(end_date))]
    
    # Metrik utama (total dan rata-rata keseluruhan)
    total_sales = filtered_df['jumlah_penjualan'].sum()
    avg_sales = filtered_df['jumlah_penjualan'].mean()
    total_revenue = filtered_df['pendapatan'].sum()
    avg_revenue = filtered_df['pendapatan'].mean()
    st.metric("Total Penjualan (Unit)", total_sales)
    st.metric("Rata-rata Penjualan Harian (Unit)", round(avg_sales, 2))
    st.metric("Total Pendapatan (Rp)", f"Rp {total_revenue:,.0f}")
    st.metric("Rata-rata Pendapatan Harian (Rp)", f"Rp {avg_revenue:,.0f}")
    
    # 1. Ringkasan penjualan setiap hari (tabel total penjualan per hari)
    st.subheader("Ringkasan Penjualan Setiap Hari")
    daily_sales = filtered_df.groupby('tanggal')[['jumlah_penjualan', 'pendapatan']].sum().reset_index()
    daily_sales = daily_sales.rename(columns={'jumlah_penjualan': 'Total Penjualan (Unit)', 'pendapatan': 'Total Pendapatan (Rp)'})
    daily_sales['Total Pendapatan (Rp)'] = daily_sales['Total Pendapatan (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(daily_sales)
    
    # 2. Ringkasan penjualan perharian (grafik tren harian)
    st.subheader("Tren Penjualan Perharian")
    if PLOTLY_AVAILABLE:
        fig = px.line(daily_sales, x='tanggal', y='Total Penjualan (Unit)', title='Total Penjualan per Hari (Unit)')
        st.plotly_chart(fig)
    else:
        st.bar_chart(daily_sales.set_index('tanggal')['Total Penjualan (Unit)'])
    
    # 3. Rata-rata barang yang dibeli per hari setiap wilayah (tabel rata-rata per hari per lokasi)
    st.subheader("Rata-rata Barang Dibeli per Hari per Wilayah")
    avg_daily_per_location = filtered_df.groupby(['lokasi', 'tanggal'])[['jumlah_penjualan', 'pendapatan']].sum().groupby('lokasi').mean().reset_index()
    avg_daily_per_location = avg_daily_per_location.rename(columns={'jumlah_penjualan': 'Rata-rata Penjualan per Hari (Unit)', 'pendapatan': 'Rata-rata Pendapatan per Hari (Rp)'})
    avg_daily_per_location['Rata-rata Pendapatan per Hari (Rp)'] = avg_daily_per_location['Rata-rata Pendapatan per Hari (Rp)'].apply(lambda x: f"Rp {x:,.0f}")
    st.dataframe(avg_daily_per_location)

elif page == "Visualisasi":
    st.title("Visualisasi Penjualan")
    
    sales_df = st.session_state.sales_df
    
    # Filter tanggal untuk interaktivitas
    start_date = st.date_input("Pilih Tanggal Mulai", value=pd.to_datetime('2023-01-01'), key="vis_start")
    end_date = st.date_input("Pilih Tanggal Akhir", value=pd.to_datetime('2023-12-31'), key="vis_end")
    filtered_df = sales_df[(sales_df['tanggal'] >= pd.to_datetime(start_date)) & (sales_df['tanggal'] <= pd.to_datetime(end_date))]
    
    # Grafik per produk (yang sudah ada)
    st.subheader("Grafik Penjualan per Produk (Unit)")
    st.bar_chart(filtered_df.groupby('produk')['jumlah_penjualan'].sum())
    
    # Grafik pendapatan per produk
    st.subheader("Grafik Pendapatan per Produk (Rp)")
    product_revenue = filtered_df.groupby('produk')['pendapatan'].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig_product_rev = px.bar(product_revenue, x='produk', y='pendapatan', title='Total Pendapatan per Produk (Rp)', color='produk')
        fig_product_rev.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_product_rev)
    else:
        st.bar_chart(product_revenue.set_index('produk')['pendapatan'])
    
    # Grafik per daerah (lokasi) - merinci jumlah penjualan per daerah
    st.subheader("Grafik Penjualan per Daerah (Unit)")
    location_sales = filtered_df.groupby('lokasi')['jumlah_penjualan'].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig_location = px.bar(location_sales, x='lokasi', y='jumlah_penjualan', title='Total Penjualan per Daerah (Unit)', color='lokasi')
        st.plotly_chart(fig_location)
    else:
        st.bar_chart(location_sales.set_index('lokasi')['jumlah_penjualan'])
    
    # Grafik pendapatan per daerah
    st.subheader("Grafik Pendapatan per Daerah (Rp)")
    location_revenue = filtered_df.groupby('lokasi')['pendapatan'].sum().reset_index()
    if PLOTLY_AVAILABLE:
        fig_location_rev = px.bar(location_revenue, x='lokasi', y='pendapatan', title='Total Pendapatan per Daerah (Rp)', color='lokasi')
        fig_location_rev.update_yaxes(tickformat=",.0f")
        st.plotly_chart(fig_location_rev)
    else:
        st.bar_chart(location_revenue.set_index('lokasi')['pendapatan'])
    
    # Peta interaktif dengan detail jumlah penjualan per daerah
    st.subheader("Peta Penjualan per Daerah (Unit)")
    # Hitung total penjualan per lokasi unik
    location_totals = filtered_df.groupby(['lokasi', 'lat', 'lon'])['jumlah_penjualan'].sum().reset_index()
    if location_totals.empty:
        st.write("Tidak ada data penjualan dalam rentang tanggal yang dipilih.")
    else:
        if PLOTLY_AVAILABLE:
            # Pastikan kolom numerik untuk Plotly
            location_totals['jumlah_penjualan'] = pd.to_numeric(location_totals['jumlah_penjualan'], errors='coerce').fillna(0)
            location_totals['lat'] = pd.to_numeric(location_totals['lat'], errors='coerce').fillna(0)
            location_totals['lon'] = pd.to_numeric(location_totals['lon'], errors='coerce').fillna(0)
            
            fig_map = px.scatter_geo(location_totals, lat="lat", lon="lon", size="jumlah_penjualan",
                                     hover_name="lokasi", hover_data={"jumlah_penjualan": True},
                                     color_discrete_sequence=["fuchsia"], scope="asia", projection="natural earth",
                                     title="Peta Penjualan per Daerah (Unit)")
            st.plotly_chart(fig_map)
        else:
            st.write("Peta tidak dapat ditampilkan karena library Plotly tidak tersedia. Gunakan grafik batang di atas sebagai alternatif.")

elif page == "Prediksi":
    st.title("Prediksi Penjualan")
    
    sales_df = st.session_state.sales_df
    
    # Latih model dan dapatkan encoder serta MSE
    model, encoder, mse = train_sales_model(sales_df)
    
    if model is None:
        st.error("Tidak ada data untuk melatih model. Tambahkan data di halaman Input Data.")
    else:
        # Input untuk prediksi
        st.subheader("Input untuk Prediksi")
        hari_input = st.number_input("Masukkan Hari dalam Bulan (1-31)", min_value=1, max_value=31, value=15)
        produk_options = sales_df['produk'].unique().tolist()  # Dinamis berdasarkan produk di data
        if not produk_options:
            st.error("Tidak ada produk dalam data. Tambahkan data penjualan terlebih dahulu.")
        else:
            produk_input = st.selectbox("Pilih Produk", options=produk_options)
            lokasi_input = st.selectbox("Pilih Lokasi", options=['Jakarta', 'Surabaya', 'Bandung'])
            
            if st.button("Prediksi Penjualan"):
                try:
                    # Encode input
                    input_encoded = encoder.transform([[produk_input, lokasi_input]])
                    encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['produk', 'lokasi']))
                    
                    # Buat input_df dengan urutan kolom yang sama seperti saat training: ['hari'] + encoded columns
                    input_df = pd.DataFrame({'hari': [hari_input]}, index=[0])
                    input_features = pd.concat([input_df, encoded_df], axis=1)
                    
                    # Prediksi
                    pred_sales = model.predict(input_features)[0]
                    
                    # Estimasi pendapatan (berdasarkan harga rata-rata produk di lokasi)
                    avg_price = sales_df[(sales_df['produk'] == produk_input) & (sales_df['lokasi'] == lokasi_input)]['harga'].mean()
                    estimated_revenue = pred_sales * avg_price if not np.isnan(avg_price) else 0
                    
                    # Tampilkan hasil rinci
                    st.subheader("Hasil Prediksi")
                    st.write(f"**Prediksi Jumlah Penjualan**: {int(pred_sales)} unit")
                    st.write(f"**Estimasi Pendapatan**: Rp {estimated_revenue:,.0f}" if estimated_revenue > 0 else "**Estimasi Pendapatan**: Data tidak cukup")
                    if mse is not None:
                        st.write(f"**Akurasi Model (MSE)**: {mse:.2f} (semakin rendah, semakin akurat)")
                    else:
                        st.write("**Akurasi Model**: Tidak dapat dihitung karena data terlalu sedikit.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi: {str(e)}")