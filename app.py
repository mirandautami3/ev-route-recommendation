import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_folium import st_folium
import folium
import pickle
import osmnx as ox
from utils.predict import predict

st.set_page_config(layout="wide")
st.title("🚗 EV Route Recommendation")

# ——————————————————————————————————————————————
# 1) Load semua assets (cached agar tidak reload tiap interaksi)
@st.cache_resource
def load_assets():
    model    = load_model("./model/dqn_ev_model.h5", compile=False)
    with open("./model/dqn_ev_graph.pkl", "rb") as f:
        G = pickle.load(f)
    df_spklu = pd.read_csv("./data/spklu.csv")
    return model, G, df_spklu

model, G, df_spklu = load_assets()

# ——————————————————————————————————————————————
# 2) Sidebar untuk input pengguna
all_connectors = (
    df_spklu["Jenis Konektor"]
    .dropna()
    .str.split(",")
    .explode()
    .str.strip()
    .str.lower()
    .unique()
    .tolist()
)
all_connectors = [conn.upper() for conn in all_connectors]

ev_type = [
    {"model": "Wuling Air EV Standard Range (17.3 Kwh)", "capacity": 17.3, "range": 200},
    {"model": "Wuling Air EV Long Range (31.9 Kwh)", "capacity": 31.9, "range": 300},
    {"model": "Wuling BinguoEV Premium Range (37.9 Kwh)", "capacity": 37.9, "range": 410},
    {"model": "Wuling BinguoEV Long Range (31.9 Kwh)", "capacity": 31.9, "range": 333},
    {"model": "Wuling Cloud EV (50.6 Kwh)", "capacity": 50.6, "range": 460}
]
options = [(ev["capacity"], ev["model"]) for ev in ev_type]

max_steps = 200
with st.sidebar:
    st.header("Input Perjalanan")
    start_address = st.text_input("📍 Alamat Awal", value="Universitas Ciputra Surabaya")
    goal_address  = st.text_input("🎯 Alamat Tujuan", value="Tunjungan Plaza")
    capacity_kwh  = st.selectbox("⚡ Kapasitas Battery EV (kWh)", options=options, format_func=lambda x: x[1] )
    soc_pct       = st.slider("🔋 SOC Awal (%)", min_value=0, max_value=100, value=20)
    connector     = st.selectbox("🔌 Jenis Connector", options=all_connectors, index=0)

    if st.button("▶️ Hitung Rute RL"):
        st.session_state.clear()
        st.session_state['run_inference'] = True

capacity_value = capacity_kwh[0]
# Validasi alamat menggunakan OSMnx sebelum inference
if st.session_state.get('run_inference', False):
    # Validasi level energi
    if soc_pct <= 20:
        st.warning("⚠️ Energi rendah (<20%). Harap isi daya terlebih dahulu.")
        st.stop()

    if not start_address or not goal_address:
        st.error("Alamat awal dan tujuan wajib diisi.")
        st.stop()
    try:
        start_point = ox.geocode(start_address)
        goal_point  = ox.geocode(goal_address)
    except Exception:
        st.error("Alamat awal atau tujuan tidak valid. Mohon cek kembali input Anda.")
        st.stop()

    with st.spinner("Mencari Rute Terbaik..."):
        if 'route_coords' not in st.session_state:
            predict = predict(
                G, model, df_spklu, soc_pct, capacity_value, start_address, goal_address,
                connector, max_steps
            )
            st.session_state.update(predict)

    # Render map dan summary jika sudah ada hasil\if 'route_coords' in st.session_state:
    route_coords   = st.session_state.get('route_coords', [])
    info           = st.session_state.get('info', {})
    visited_spklu = st.session_state.get('visited_spklu', {})
    total_distance = st.session_state.get('total_distance', 0.0)
    total_energy   = st.session_state.get('total_energy', 0.0)
    status         = st.session_state.get('status', '')
    capacity_kwh   = st.session_state.get('capacity_kwh', 0.0)
    soc_pct        = st.session_state.get('soc_pct', 0.0)

    if visited_spklu:
        visited_name = [spklu['spklu_name'] for spklu in visited_spklu]
        spklu_name = ', '.join(visited_name)
    else:
        spklu_name = '-'

    if len(route_coords) > 1:
        # Buat Map Folium
        m = folium.Map(location=route_coords[0], zoom_start=13, tiles="OpenStreetMap")
        folium.PolyLine(route_coords, weight=4, opacity=0.8).add_to(m)
        folium.Marker(route_coords[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(route_coords[-1], popup="Goal", icon=folium.Icon(color="red")).add_to(m)
        for sp in info.get("visited_spklu", []):
            node = sp['node'] if isinstance(sp, dict) else sp
            name = sp.get('spklu_name', f"SPKLU {node}") if isinstance(sp, dict) else G.nodes[node].get('spklu_name',
                                                                                                        f"SPKLU {node}")
            lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
            folium.Marker((lat, lon), popup=name,
                          icon=folium.Icon(color="orange", icon="bolt", prefix="fa")).add_to(m)

        # Container dengan 2 kolom: peta di kiri, ringkasan di kanan
        with st.container():
            left, right = st.columns([3, 1], gap="medium")

            with left:
                st.markdown("### 🗺️ Hasil Rute", unsafe_allow_html=True)
                # Wrapper div untuk max-height + scroll
                st.markdown(
                    '<div style="max-height:400px; overflow-y:auto; border:1px solid #ddd;">',
                    unsafe_allow_html=True
                )
                st_folium(m, use_container_width=True, height=500)
                st.markdown('</div>', unsafe_allow_html=True)

            with right:
                st.markdown(
                    """
                    <div style="
                        background:#f0f2f6;
                        padding:12px;
                        border-radius:8px;
                        box-shadow:0 2px 4px rgba(0,0,0,0.1);
                        margin-top:60px;
                        color: black;
                        margin-bottom:15px;
                    ">
                      <h4 style="margin:0 0 0 0;">📋 Ringkasan Perjalanan</h4>
                    """,
                    unsafe_allow_html=True
                )
                # metrics with smaller text
                st.markdown(f'<p style="margin:3px 0; font-size:16px;"><b>Start:</b> {start_address}</p>',
                            unsafe_allow_html=True)
                st.markdown(f'<p style="margin:3px 0; font-size:16px;"><b>Goal:</b> {goal_address}</p>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin:3px 0; font-size:16px;"><b>Total Distance (km):</b> {total_distance:.2f}</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin:3px 0; font-size:16px;"><b>Total Energy (kWh):</b> {total_energy:.2f}</p>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<p style="margin:3px 0; font-size:16px;"><b>Visited SPKLU:</b> {spklu_name}</p>',
                    unsafe_allow_html=True)
                st.markdown(f'<p style="margin:3px 0; font-size:16px;"><b>Energy Status:</b> {status}</p>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# run python -m streamlit run app.py