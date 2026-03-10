import streamlit as st
import leafmap.foliumap as leafmap
import rasterio
import rasterio.mask
from pyproj import Transformer
import concurrent.futures
from streamlit_folium import st_folium
import io
import zipfile
import time
from shapely.geometry import shape, mapping
from shapely.ops import transform

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

if "aoi_zip_buffer" not in st.session_state:
    st.session_state["aoi_zip_buffer"] = None

st.title("🌲 Prostorová extrapolace parametrů NIL3")
st.markdown("Interaktivní vizualizace mapových vrstev Národní inventarizace lesů vzniklých natrénováním ensemble modelů strojového učení (prediktory: Výškový model lesa 2018-2019, odrazivosti Sentinel-2 2019).")
st.markdown("---")

# ==========================================
# 2. Data Processing Functions
# ==========================================
def fetch_pixel_value(url, x, y):
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

def process_single_layer(task, geom_mapping, env_kwargs):
    """Worker function for parallel raster clipping."""
    key, stat, base_url = task
    url = f"{base_url}masked_predicted_{key}_10m_{stat}_cog.tif"
    vsi_url = f"/vsicurl/{url}"
    try:
        with rasterio.Env(**env_kwargs):
            with rasterio.open(vsi_url) as src:
                out_image, out_transform = rasterio.mask.mask(src, geom_mapping, crop=True)
                out_meta = src.meta.copy()

                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "compress": "deflate"
                })

                mem_file = io.BytesIO()
                with rasterio.open(mem_file, "w", **out_meta) as dest:
                    dest.write(out_image)
                
                return f"NIL3_{key}_{stat}.tif", mem_file.getvalue()
    except Exception as e:
        return None, str(e)

def clip_and_zip_aoi(geojson_geometry, targets, base_url, progress_bar, status_text):
    # Transform geometry from WGS84 to EPSG:32633 for metric area calculation
    geom = shape(geojson_geometry)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
    geom_proj = transform(transformer.transform, geom)
    
    # Area limit validation (100 km2)
    area_km2 = geom_proj.area / 1_000_000.0
    if area_km2 > 100.0:
        return None, f"Zvolené území ({area_km2:.1f} km²) překračuje limit 100 km². Zmenšete polygon."
    
    geom_mapping = [mapping(geom_proj)]
    zip_buffer = io.BytesIO()
    
    env_kwargs = {
        'GDAL_HTTP_FOLLOWREDIRECTS': 'YES',
        'GDAL_HTTP_USERAGENT': 'Mozilla/5.0',
        'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
        'VSI_CACHE': 'TRUE'
    }

    # Prepare tasks for ThreadPool
    tasks = [(k, stat, base_url) for k in targets.keys() for stat in ["mean", "cv"]]
    total_tasks = len(tasks)
    completed = 0
    results_data = []

    # Execute clipping in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(process_single_layer, task, geom_mapping, env_kwargs): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            file_name, data = future.result()
            if file_name is not None:
                results_data.append((file_name, data))
            
            completed += 1
            progress_bar.progress(completed / total_tasks)
            status_text.text(f"Zpracováno {completed}/{total_tasks} vrstev (Plocha: {area_km2:.1f} km²)...")

    # Package into ZIP
    status_text.text("Komprimuji data do ZIP archivu...")
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in results_data:
            zip_file.writestr(file_name, data)

    zip_buffer.seek(0)
    return zip_buffer, None

# ==========================================
# 3. Parameters & UI (Sidebar)
# ==========================================
TARGETS = {
    "above_st_b": {"name": "Nadzemní biomasa", "unit": "t/ha", "max_val": 500, "max_cv": 80, "rrmse": 42.36},
    "gldsity_vo": {"name": "Objem hroubí", "unit": "m³/ha", "max_val": 800, "max_cv": 80, "rrmse": 40.68},
    "h_plot": {"name": "Výška porostu", "unit": "m", "max_val": 45, "max_cv": 50, "rrmse": 18.25},
    "age_plot": {"name": "Věk porostu", "unit": "let", "max_val": 160, "max_cv": 80, "rrmse": 35.84},
    "dbh_plot": {"name": "Výčetní tloušťka", "unit": "mm", "max_val": 700, "max_cv": 80, "rrmse": 28.78},
    "g13_ldsity": {"name": "Kruhová základna", "unit": "m²/ha", "max_val": 60, "max_cv": 80, "rrmse": 38.48}
}

HF_BASE_URL = "https://huggingface.co/datasets/lukespetr/NIL_retrieval/resolve/main/"

with st.sidebar:
    st.header("⚙️ Nastavení vrstvy")
    selected_key = st.selectbox("🎯 Výběr parametru NIL3:", options=list(TARGETS.keys()), format_func=lambda x: TARGETS[x]["name"])
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
    st.info("💡 **Tip:** Nakreslete v mapě polygon pomocí panelu nástrojů na levé straně pro hromadné stažení dat, nebo klikněte do mapy pro zobrazení lokálního profilu.")

suffix = "mean" if "Průměr" in map_mode else "cv"
cog_url = f"{HF_BASE_URL}masked_predicted_{selected_key}_10m_{suffix}_cog.tif"

vmax = TARGETS[selected_key]["max_val"] if suffix == "mean" else TARGETS[selected_key]["max_cv"]
palette = "viridis" if suffix == "mean" else "magma"
legend_title = f"{TARGETS[selected_key]['name']}" if suffix == "mean" else f"Nejistota CV (%)"

# ==========================================
# 4. Map Initialization
# ==========================================
m = leafmap.Map(
    center=[49.19, 16.60], 
    zoom=10, 
    draw_control=True, 
    measure_control=False
)

m.add_basemap(basemap_options[selected_basemap])

with st.spinner(f"🛰️ Načítám vrstvu: {TARGETS[selected_key]['name']}..."):
    m.add_cog_layer(
        url=cog_url,
        name=f"{TARGETS[selected_key]['name']} ({suffix.upper()})",
        palette=palette,
        rescale=f"1,{vmax}", 
        transparent_bg=True,
        nodata=0,
        opacity=layer_opacity
    )
    m.add_colormap(cmap=palette, vmin=1, vmax=vmax, label=legend_title, position="topright")

    if show_ndsm_vector:
        pmtiles_url = "https://pub-ddf1e6086fe44d9dbcdf57d66b64fef0.r2.dev/nDSM_change_NIL3_fixed.pmtiles"
        maplibre_style = {
            "version": 8,
            "sources": {"ndsm_source": {"type": "vector", "url": f"pmtiles://{pmtiles_url}"}},
            "layers": [{
                "id": "ndsm_polygons",
                "type": "fill",
                "source": "ndsm_source",
                "source-layer": "nDSM_change_NIL3_fixed — NIL3_polygons", 
                "paint": {
                    "fill-color": "#ef5350",
                    "fill-opacity": 0.5, 
                    "fill-outline-color": "#b71c1c"
                }
            }]
        }
        m.add_pmtiles(
            url=pmtiles_url, 
            name="Změny nDSM", 
            style=maplibre_style, 
            overlay=True, 
            control=True
        )

# === KLÍČOVÝ HACK PRO ZACHOVÁNÍ POZICE MAPY ===
# 1. Odstraníme automatický bounding box, který si folium vytvořil.
keys_to_remove = [k for k in m._children.keys() if "fitbounds" in k.lower() or "fit_bounds" in k.lower()]
for k in keys_to_remove:
    del m._children[k]

# 2. Resetujeme skrytě změněné souřadnice mapy. Leafmap je při importu vrstvy 
# posunul na střed rastru, což by způsobilo odskok mapy při každém překreslení ve Streamlitu.
m.location = [49.19, 16.60]
m.options['zoom'] = 10

with st.spinner("🗺️ Vykresluji interaktivní mapu..."):
    map_output = st_folium(m, key="nil3_main_map", width=1500, height=650, returned_objects=["last_clicked", "last_active_drawing"])

# ==========================================
# 5. Bulk Export (AOI Download)
# ==========================================
st.subheader("📥 Export dat pro zájmové území (AOI)")
if map_output and map_output.get("last_active_drawing"):
    geom = map_output["last_active_drawing"]["geometry"]
    st.success("Byl detekován polygon. Nyní můžete extrahovat mapové výstupy (12 vrstev) do formátu GeoTIFF.")
    
    if st.button("Zpracovat data pro vybraný polygon", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        buffer, error_msg = clip_and_zip_aoi(geom, TARGETS, HF_BASE_URL, progress_bar, status_text)
        
        if error_msg:
            st.error(error_msg)
            progress_bar.empty()
            status_text.empty()
        else:
            st.session_state["aoi_zip_buffer"] = buffer
            status_text.success("✅ Extrakce úspěšně dokončena. Soubor je připraven ke stažení.")
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
            
    if st.session_state.get("aoi_zip_buffer") is not None:
        st.download_button(
            label="Stáhnout připravený ZIP archiv",
            data=st.session_state["aoi_zip_buffer"],
            file_name="NIL3_AOI_export.zip",
            mime="application/zip",
            type="primary"
        )
else:
    st.info("Pro hromadný export vrstev nakreslete v mapě libovolný polygon (nástroj 'Draw a polygon' na levé straně mapy).")

st.markdown("---")

# ==========================================
# 6. Interactive Pixel Querying
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
                    std_error = (mean_val * v["rrmse"]) / 100
                    lower_bound = max(0, mean_val - std_error)
                    upper_bound = mean_val + std_error
                    precision = 0 if v['unit'] in ['t/ha', 'm³/ha', 'mm', 'let'] else 1
                    
                    st.metric(
                        label=v["name"],
                        value=f"{mean_val:.{precision}f} ± {std_error:.{precision}f} {v['unit']}",
                        delta=f"Nejistota modelu: {cv_val:.1f} % CV" if cv_val else None,
                        delta_color="off"
                    )
                    st.caption(f"Interval (1σ): {lower_bound:.1f} až {upper_bound:.1f} {v['unit']}")
                else:
                    st.metric(
                        label=v["name"], 
                        value="Mimo lesní masku", 
                        delta="Žádná data"
                    )
else:
    st.info("👆 Klikněte do mapy na libovolný zalesněný pixel pro zobrazení lokálních parametrů.")
