import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
from pyproj import Transformer
import concurrent.futures
from streamlit_folium import st_folium

# ==========================================
# 1. Konfigurace aplikace a Vlastní CSS
# ==========================================
st.set_page_config(layout="wide", page_title="NIL3 Portál", page_icon="🌲")

st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 2px 4px 15px rgba(0,0,0,0.1);
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #1e1e1e;
            border: 1px solid #333;
        }
    }
    h1 {
        color: #2e7d32;
        padding-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🌲 Prostorová extrapolace parametrů NIL3")
st.markdown("Interaktivní vizualizace mapových vrstev Národní inventarizace lesů vzniklých natrénováním ensemble modelů strojového učení (prediktory: Výškový model lesa 2018-2019, odrazivosti Sentinel-2 2019).")
st.markdown("---")

# ==========================================
# 2. Funkce pro stahování dat
# ==========================================
def fetch_pixel_value(url, x, y):
    """Stáhne hodnotu konkrétního pixelu z COG souboru na Hugging Face."""
    env_kwargs = {
        'GDAL_HTTP_FOLLOWREDIRECTS': 'YES',
        'GDAL_HTTP_USERAGENT': 'Mozilla/5.0',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif'
    }
    try:
        with rasterio.Env(**env_kwargs):
            vsi_url = f"/vsicurl/{url}" if url.startswith("http") else url
            with rasterio.open(vsi_url) as src:
                row, col = src.index(x, y)
                val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                return float(val) if val > 0 else None
    except Exception:
        return None

# ==========================================
# 3. Parametry a UI (Postranní panel)
# ==========================================
TARGETS = {
    "above_st_b": {"name": "Nadzemní biomasa (t/ha)", "max_val": 500, "max_cv": 80},
    "gldsity_vo": {"name": "Objem hroubí (m³/ha)", "max_val": 800, "max_cv": 80},
    "h_plot":     {"name": "Výška porostu (m)", "max_val": 45, "max_cv": 50},
    "age_plot":   {"name": "Věk porostu (roky)", "max_val": 160, "max_cv": 80},
    "dbh_plot":   {"name": "Výčetní tloušťka (mm)", "max_val": 700, "max_cv": 80},
    "g13_ldsity": {"name": "Kruhová základna (m²/ha)", "max_val": 60, "max_cv": 80}
}

HF_BASE_URL = "https://huggingface.co/datasets/lukespetr/NIL_retrieval/resolve/main/"

with st.sidebar:
    st.header("⚙️ Nastavení vrstvy")
    
    selected_key = st.selectbox(
        "🎯 Výběr parametru NIL3:", 
        options=list(TARGETS.keys()), 
        format_func=lambda x: TARGETS[x]["name"]
    )

    map_mode = st.radio("📊 Typ mapy:", ["Průměrný odhad", "Nejistota odhadu (CV %)"])

    st.markdown("---")
    st.header("🗺️ Nastavení zobrazení")

    basemap_options = {
        "Satelitní (Google)": "HYBRID",
        "OpenStreetMap": "OpenStreetMap",
        "Topo Mapa (OpenTopoMap)": "OpenTopoMap",
        "Terénní (Google)": "TERRAIN"
    }
    selected_basemap = st.selectbox("🌍 Základní mapa:", options=list(basemap_options.keys()))

    layer_opacity = st.slider("👁️ Průhlednost rastrové vrstvy:", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
    
    st.markdown("---")
    st.header("📐 Vektorové vrstvy")
    show_ndsm_vector = st.checkbox("Zobrazit změny nDSM (polygony)", value=False)

    st.markdown("---")
    st.info("💡 **Tip:** Kliknutím kamkoliv do lesní plochy v mapě získáte okamžitou extrakci všech 12 parametrů pro daný pixel.")

suffix = "mean" if "Průměr" in map_mode else "cv"
cog_url = f"{HF_BASE_URL}masked_predicted_{selected_key}_10m_{suffix}_cog.tif"

vmax = TARGETS[selected_key]["max_val"] if suffix == "mean" else TARGETS[selected_key]["max_cv"]
palette = "viridis" if suffix == "mean" else "magma"
legend_title = f"{TARGETS[selected_key]['name']}" if suffix == "mean" else f"Nejistota CV (%)"

# ==========================================
# 4. Inicializace a vykreslení mapy
# ==========================================
m = leafmap.Map(center=[49.19, 16.60], zoom=10, draw_control=False, measure_control=False)

m.add_basemap(basemap_options[selected_basemap])

with st.spinner(f"🛰️ Načítám vrstvu: {TARGETS[selected_key]['name']}..."):
    # 1. Přidání rastrové COG vrstvy
    m.add_cog_layer(
        url=cog_url,
        name=f"{TARGETS[selected_key]['name']} ({suffix.upper()})",
        palette=palette,
        rescale=f"1,{vmax}", 
        transparent_bg=True,
        nodata=0,
        opacity=layer_opacity
    )
    
    m.add_colormap(
        cmap=palette, 
        vmin=1, 
        vmax=vmax, 
        label=legend_title,
        position="topright" 
    )

# 2. Přidání vektorové PMTiles vrstvy (pokud je aktivována)
    if show_ndsm_vector:
        # Replace the URL below with your actual R2.dev subdomain URL
        base_url = "https://pub-ddf1e6086fe44d9dbcdf57d66b64fef0.r2.dev/nDSM_change_NIL3.pmtiles"
        pmtiles_url = f"{base_url}?nocache={int(time.time())}"
        
        maplibre_style = {
            "version": 8,
            "sources": {
                "ndsm_source": {
                    "type": "vector",
                    "url": f"pmtiles://{pmtiles_url}"
                }
            },
            "layers": [
                {
                    "id": "ndsm_polygons",
                    "type": "fill",
                    "source": "ndsm_source",
                    "source-layer": "NIL3_Polygons", 
                    "paint": {
                        "fill-color": "#ef5350",
                        "fill-opacity": 0.4,
                        "fill-outline-color": "#d32f2f"
                    }
                }
            ]
        }

        m.add_pmtiles(
            url=pmtiles_url,
            name="Změny nDSM",
            style=maplibre_style,
            overlay=True,
            control=True
        )

# Vykreslení do Streamlitu
with st.spinner("🗺️ Vykresluji interaktivní mapu..."):
    map_output = st_folium(m, width=1500, height=650, returned_objects=["last_clicked"])

# Patička
col1, col2 = st.columns([3, 1])
with col1:
    st.caption(f"**Zdrojový soubor:** `{cog_url.split('/')[-1]}`")
with col2:
    st.link_button("📥 Stáhnout zobrazený TIFF", f"{cog_url}?download=true", use_container_width=True)

st.markdown("---")

# ==========================================
# 5. Interaktivní dotazování na pixely (Dashboard)
# ==========================================
if map_output and map_output.get("last_clicked"):
    lat = map_output["last_clicked"]["lat"]
    lon = map_output["last_clicked"]["lng"]
    
    st.subheader("📍 Profil lesa pro vybranou lokalitu")
    st.caption(f"**GPS souřadnice:** {lat:.5f} N, {lon:.5f} E")
    st.write("") 
    
    with st.spinner("⚡ Extrahuji parametry ze všech 12 predikčních vrstev v reálném čase..."):
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        x, y = transformer.transform(lon, lat)
        
        queries = []
        for k in TARGETS.keys():
            queries.append((k, "mean", f"{HF_BASE_URL}masked_predicted_{k}_10m_mean_cog.tif"))
            queries.append((k, "cv", f"{HF_BASE_URL}masked_predicted_{k}_10m_cv_cog.tif"))
        
        results = {k: {"mean": None, "cv": None} for k in TARGETS.keys()}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            future_to_key = {executor.submit(fetch_pixel_value, q[2], x, y): q for q in queries}
            for future in concurrent.futures.as_completed(future_to_key):
                var_key, stat_type, _ = future_to_key[future]
                results[var_key][stat_type] = future.result()

        cols = st.columns(3)
        for i, (k, v) in enumerate(TARGETS.items()):
            mean_val = results[k]["mean"]
            cv_val = results[k]["cv"]
            
            with cols[i % 3]:
                if mean_val is not None:
                    st.metric(
                        label=v["name"],
                        value=f"{mean_val:.1f}",
                        delta=f"Nejistota: ± {cv_val:.1f} % CV" if cv_val is not None else "N/A",
                        delta_color="off" 
                    )
                else:
                    st.metric(label=v["name"], value="Mimo lesní masku", delta="Žádná data")
else:
    st.info("👆 Klikněte do mapy na libovolný zalesněný pixel pro zobrazení lokálních parametrů.")








