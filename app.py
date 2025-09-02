import streamlit as st
try:
    st.set_page_config(page_title='PRADAR · Dynamic Pricing VR', page_icon='images/logo_pradar_main.png', layout='wide')
except Exception:
    pass
st.image('images/logo_pradar_main.png', width=220)

import os
from pathlib import Path
import json
import datetime as dt
import numpy as np
import pandas as pd
import duckdb
import streamlit as st
import logging
import unicodedata, re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PRADAR")

st.set_page_config(
    page_title="PRADAR - Price Radar",
    page_icon="💶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------- CSS (sidebar ancha + botones con estado) -----------
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 460px !important; min-width: 460px !important; }
section[data-testid="stSidebar"] .block-container { padding-top: 0.25rem; }
.main .block-container { padding-top: 0.5rem; }
.stSelectbox, .stMultiSelect, .stSlider { width: 100% !important; }
.btn-prof { padding: 14px 16px; border-radius: 14px; border: 1px solid #e5e7eb; font-weight: 700; white-space: pre-line; width: 100%; text-align: left; }
.btn-C  { background:#eff6ff; border-color:#bfdbfe; color:#1e3a8a; }
.btn-N  { background:#ecfdf5; border-color:#a7f3d0; color:#065f46; }
.btn-R  { background:#fff7ed; border-color:#fed7aa; color:#9a3412; }
.btn-selected { outline: 3px solid #111827; }
.smallhelp { font-size: 12px; color: #6b7280; margin-top: -6px; }
.h1-center { text-align:center; margin: 0; padding: 0; }
.sub-center { text-align:center; color:#6b7280; margin-top: 2px; }
section[data-testid="stSidebar"] .stMarkdown, 
section[data-testid="stSidebar"] .stSelectbox, 
section[data-testid="stSidebar"] .stSlider { margin-bottom: 0.4rem; }
</style>
""", unsafe_allow_html=True)

# --------------------  Paths  --------------------
def discover_root() -> Path:
    cands = []
    if os.getenv("PRADAR_ROOT"): cands.append(Path(os.environ["PRADAR_ROOT"]))
    cands += [Path.cwd(),
              Path.home()/ "Desktop/Pradar/PRADAR V3",
              Path.home()/ "Desktop/Pradar/Pradar V3"]
    for p in cands:
        if (p/"data/processed").exists(): return p
    return Path.cwd()

PROJECT_ROOT = discover_root()
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
ASSETS_DIR   = PROJECT_ROOT / "assets"

# --------------------  Mappings  --------------------
PROFILE_TO_COL = {"Conservador": "price_C", "Normal": "price_N", "Riesgo": "price_R"}
PROFILE_EMOJI  = {"Conservador":"🔵", "Normal":"🟢", "Riesgo":"🟠"}
PROFILE_SUB    = {
    "Conservador": "Más ocupación, menor riesgo",
    "Normal":      "Equilibrio entre noches e ingresos",
    "Riesgo":      "Maximiza ingresos, acepta riesgo"
}

AMENITY_LABELS = {
    "amen_pool": "Piscina", "amen_ac": "Aire acondicionado", "amen_terrace": "Terraza",
    "amen_sea_view": "Vista al mar", "amen_beachfront": "Frente playa", "amen_parking": "Parking",
    "amen_garden": "Jardín", "amen_hot_tub": "Jacuzzi", "amen_heating": "Calefacción",
    "amen_fireplace": "Chimenea", "amen_elevator": "Ascensor"
}

# Municipios y localidades de Mallorca (fallback robusto)
MALLORCA_FALLBACK_LOCS = sorted(list({
    "Palma", "Calvià", "Marratxí", "Llucmajor", "Andratx", "Sóller", "Deià", "Valldemossa",
    "Banyalbufar", "Estellencs", "Puigpunyent", "Esporles",
    "Inca", "Binissalem", "Lloseta", "Alaró", "Selva", "Mancor de la Vall", "Búger", "Campanet",
    "Manacor", "Felanitx", "Sant Llorenç des Cardassar", "Son Servera", "Santanyí", "Ses Salines",
    "Pollença", "Alcúdia", "Sa Pobla", "Muro", "Santa Margalida", "Artà", "Capdepera",
    "Campos", "Porreres", "Vilafranca de Bonany", "Petra", "Sineu", "Costitx", "Lloret de Vistalegre",
    "Montuïri", "Algaida", "Consell", "Santa Maria del Camí", "Bunyola", "Llubí", "Sencelles"
}))

# --------------------  Helpers  --------------------
def group_property_type(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.lower()
    apto = ["apartment","apartamento","condo","condominium","flat","loft","studio"]
    house= ["house","casa","villa","chalet","townhouse","finca","cottage","bungalow"]
    priv = ["private room","habitación","habitacion","room in", "private-room", "hab. privada"]
    htl  = ["hotel","serviced","aparthotel","hostel","guesthouse","boutique"]
    def _map(v:str):
        if not v or v=="<na>": return "Otros"
        if any(k in v for k in house): return "Casa/Villa"
        if any(k in v for k in apto):  return "Apartamento"
        if any(k in v for k in priv):  return "Hab. privada"
        if any(k in v for k in htl):   return "Hotel/Serviced"
        return "Otros"
    return x.map(_map)

def _clean_text_series(s: pd.Series) -> pd.Series:
    sv = s.astype("string").str.strip()
    sv = sv.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    return sv

def _normalize_key_str(v: str | None) -> str | None:
    if v is None or (isinstance(v, float) and np.isnan(v)): return None
    s = str(v).strip()
    if not s: return None
    s = re.sub(r"\s+", " ", s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    return s.lower()

@st.cache_data(show_spinner=False, ttl=3600)
def parquet_columns(fp: Path):
    try:
        import pyarrow.parquet as pq
        return pq.ParquetFile(str(fp)).schema.names
    except Exception:
        pass
    try:
        con = duckdb.connect(database=":memory:")
        path_str = str(fp).replace("'", "''")
        df0 = con.execute(f"SELECT * FROM read_parquet('{path_str}') LIMIT 0").df()
        con.close()
        return df0.columns.tolist()
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def load_pricing_source():
    variant = "grid"
    cfg = MODELS_DIR / "pricing_config.json"
    if cfg.exists():
        try:
            variant = json.loads(cfg.read_text()).get("active_variant","grid")
        except Exception:
            pass
    cands = [
        DATA_PROC / "app_prices_active.parquet",
        DATA_PROC / ("df_prices.parquet" if variant=="grid" else "df_prices_eta.parquet"),
        DATA_PROC / "df_prices.parquet",
        DATA_PROC / "df_prices_eta.parquet",
    ]
    src = next((p for p in cands if p.exists()), None)
    if not src:
        st.error("No se encontraron archivos de precios en data/processed.")
        st.stop()

    cols = parquet_columns(src)
    name_map = {}
    if "price_conservador" in cols: name_map["price_conservador"]="price_C"
    if "price_normal"      in cols: name_map["price_normal"]="price_N"
    if "price_riesgo"      in cols: name_map["price_riesgo"]="price_R"

    lid_cands  = [c for c in ["listing_id","listing","property_id","id"] if c in cols]
    date_cands = [c for c in ["date","ds","calendar_date","fecha"] if c in cols]
    return {"path": src,
            "lid":  lid_cands[0]  if lid_cands  else cols[0],
            "dcol": date_cands[0] if date_cands else cols[1] if len(cols)>1 else cols[0],
            "name_map": name_map,
            "all_cols": cols}

@st.cache_data(show_spinner=False, ttl=3600)
def load_location_index():
    cands = [
        MODELS_DIR / "location_index.json",
        MODELS_DIR / "geo_locations.json",
        MODELS_DIR / "locations.json",
        ASSETS_DIR / "location_index.json",
        ASSETS_DIR / "geo_locations.json",
        ASSETS_DIR / "locations.json",
    ]
    for fp in cands:
        if fp.exists():
            try:
                data = json.loads(fp.read_text())
                if isinstance(data, dict):
                    for key in ["locations","values","items","locs","lista"]:
                        if key in data and isinstance(data[key], list):
                            return [str(x).strip() for x in data[key] if str(x).strip()]
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except Exception as e:
                logger.warning(f"No se pudo leer {fp.name}: {e}")
    return []

@st.cache_data(show_spinner=False, ttl=3600)
def load_meta():
    desired = [
        "listing_id","listing","property_id","id",
        "city","municipality","municipio","pueblo","town","locality","localidad",
        "neighbourhood","neighborhood","neighbourhood_cleansed","neighborhood_cleansed",
        "neighbourhood_group_cleansed","district","area","zone","borough","region",
        "address","street",
        "property_type","bedrooms","bathrooms","accommodates",
        "minimum_nights","min_nights","host_is_superhost",
        "review_scores_rating","number_of_reviews","reviews_count","n_reviews","review_count",
        *AMENITY_LABELS.keys()
    ]
    df = pd.DataFrame()
    for fn in ["app_meta.parquet","df_features_clean.parquet","df_features.parquet"]:
        fp = DATA_PROC / fn
        if fp.exists():
            try:
                use = [c for c in desired if c in parquet_columns(fp)]
                df = pd.read_parquet(fp, columns=use).copy()
                break
            except Exception as e:
                logger.warning(f"Meta load error {fn}: {e}")
    if df.empty:
        st.error("No se pudieron cargar metadatos.")
        st.stop()

    id_cands = [c for c in ["listing_id","listing","property_id","id"] if c in df.columns]
    if id_cands and id_cands[0]!="listing_id":
        df = df.rename(columns={id_cands[0]:"listing_id"})
    df["listing_id"] = df["listing_id"].astype(str)
    df = df.drop_duplicates("listing_id")

    ucands = ["city","municipality","municipio","pueblo","town","locality","localidad",
              "neighbourhood","neighborhood","neighbourhood_cleansed","neighborhood_cleansed",
              "neighbourhood_group_cleansed","district","area","zone","borough","region",
              "address","street"]
    for c in ucands:
        if c in df.columns:
            df[c] = _clean_text_series(df[c])

    def coalesce(cols):
        s = pd.Series(pd.NA, index=df.index, dtype="object")
        for c in cols:
            if c in df.columns:
                v = df[c]
                s = s.fillna(v)
        return s

    def norm_title(s: pd.Series) -> pd.Series:
        ss = s.astype("string")
        ss = ss.where(ss.isna(), ss.str.strip())
        ss = ss.where(ss.isna(), ss.str.replace(r"\s+", " ", regex=True))
        return ss.where(ss.isna(), ss.str.title())

    # niveles legibles
    df["loc_city"]  = norm_title(coalesce(["city","municipality","municipio","pueblo","town","locality","localidad"]))
    df["loc_muni"]  = norm_title(coalesce(["municipality","municipio","city","pueblo","town","locality","localidad"]))
    df["loc_neigh"] = norm_title(coalesce(["neighbourhood","neighborhood","neighbourhood_cleansed","neighborhood_cleansed",
                                           "neighbourhood_group_cleansed","district","area","zone","borough","region"]))
    df["location_any"] = norm_title(coalesce([
        "neighbourhood","neighborhood","neighbourhood_cleansed","neighborhood_cleansed",
        "neighbourhood_group_cleansed","district","area","zone","borough","region",
        "municipality","municipio","city","pueblo","town","locality","localidad",
        "address","street"
    ]))

    # claves canónicas
    def key_series(s: pd.Series) -> pd.Series:
        return s.astype("string").map(_normalize_key_str)

    df["loc_city_key"]  = key_series(df["loc_city"])
    df["loc_muni_key"]  = key_series(df["loc_muni"])
    df["loc_neigh_key"] = key_series(df["loc_neigh"])
    df["location_any_key"] = key_series(df["location_any"])

    df["location_norm_key"] = coalesce(["loc_neigh_key","loc_muni_key","loc_city_key","location_any_key"])
    df["location_display"]  = norm_title(coalesce(["loc_neigh","loc_muni","loc_city","location_any"]))

    # Prop group
    df["prop_group"] = group_property_type(df.get("property_type","Otros"))
    cat_order = ["Casa/Villa","Apartamento","Hab. privada","Otros","Hotel/Serviced"]
    df["prop_group"] = pd.Categorical(df["prop_group"], categories=cat_order, ordered=True)

    for c in ["bedrooms","bathrooms","accommodates","minimum_nights","min_nights","review_scores_rating"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    rev_cands = [c for c in ["number_of_reviews","reviews_count","n_reviews","review_count"] if c in df.columns]
    if rev_cands:
        df["n_reviews"] = pd.to_numeric(df[rev_cands[0]], errors="coerce").fillna(0).astype(int)
    else:
        df["n_reviews"] = 0

    for a in [c for c in df.columns if c.startswith("amen_")]:
        s = df[a]
        if pd.api.types.is_bool_dtype(s):
            df[a] = s.fillna(False).astype("int8")
        elif pd.api.types.is_numeric_dtype(s):
            df[a] = (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype("int8")
        else:
            df[a] = s.astype("string").str.lower().isin(["true","1","yes","si","sí","y","t"]).astype("int8")
    return df

def build_location_options(meta: pd.DataFrame):
    if "location_display" in meta.columns:
        opts = meta["location_display"].dropna().astype("string")
        opts = opts[opts.str.len() > 0]
        opts = sorted(pd.Series(opts.unique(), dtype="string").tolist())
        if opts: return opts
    return MALLORCA_FALLBACK_LOCS

# --------- Cache de precios por ventana ---------
@st.cache_data(show_spinner=False, ttl=600, max_entries=12)
def read_prices_window(src_path: Path, lid_col: str, date_col: str,
                       price_col: str, d0: dt.date, d1: dt.date) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    src_esc = str(src_path).replace("'", "''")
    sql = f"""
      SELECT CAST({date_col} AS DATE) AS date,
             CAST({lid_col}  AS VARCHAR) AS lid,
             CAST({price_col} AS DOUBLE)  AS price
      FROM read_parquet('{src_esc}')
      WHERE {date_col} BETWEEN DATE '{d0.isoformat()}' AND DATE '{d1.isoformat()}'
        AND {price_col} IS NOT NULL AND {price_col} > 0
    """
    df = con.execute(sql).df()
    con.close()
    return df[(df["price"]>0) & (df["price"]<100000)]

@st.cache_data(show_spinner=False, ttl=600, max_entries=24)
def available_lids_in_window(src_path: Path, lid_col: str, date_col: str,
                             price_col: str, d0: dt.date, d1: dt.date) -> list[str]:
    df = read_prices_window(src_path, lid_col, date_col, price_col, d0, d1)
    return sorted(df["lid"].astype(str).unique().tolist())

# -------------------- Similaridad y fallback --------------------
def _as_float_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()

def _range_closeness(series: pd.Series, lo: float, hi: float) -> np.ndarray:
    s = _as_float_array(series)
    lo = float(lo); hi = float(hi)
    inside = (s >= lo) & (s <= hi)
    dist = np.where(s < lo, lo - s, np.where(s > hi, s - hi, 0.0))
    clos = np.where(inside, 1.0, 1.0/(1.0 + dist))
    return np.where(np.isnan(s), 0.6, clos)

def _rating_closeness(series: pd.Series, min_req: float) -> np.ndarray:
    s = _as_float_array(series)
    if np.isfinite(s).any():
        max_val = np.nanmax(s)
        if max_val > 5.1:
            s = s / 20.0
    else:
        s = s  # all NaN
    base = np.where(np.isnan(s), 0.6,
                    np.where(s >= min_req, 1.0, np.maximum(0.3, (s / (min_req if min_req > 0 else 1e-9)) * 0.85)))
    return base

def _reviews_closeness(series: pd.Series, min_req: int) -> np.ndarray:
    s = _as_float_array(series)
    s = np.nan_to_num(s, nan=0.0)
    diff = np.maximum(0.0, float(min_req) - s)
    return 1.0 / (1.0 + diff/10.0)

def knn_candidate_ids(meta: pd.DataFrame,
                      loc_choice: str,
                      sel_prop: str,
                      rng_acc: tuple[int,int],
                      rng_bed: tuple[int,int],
                      rng_bath: tuple[int,int],
                      rating_min: float,
                      n_reviews_min: int,
                      sel_amens: list[str],
                      top_k: int = 400) -> list[str]:
    m = meta.copy()

    # Localidad (exacta -> contiene -> resto), evitando NA
    disp = m["location_display"].fillna("").astype("string")
    disp_key = disp.map(_normalize_key_str)
    choice_key = _normalize_key_str(loc_choice) if loc_choice and loc_choice != "— Todos —" else None
    if choice_key:
        eq = (disp_key == choice_key).fillna(False).to_numpy()
        contains = disp_key.str.contains(re.escape(choice_key), na=False).to_numpy()
        loc_clos = np.where(eq, 1.0, np.where(contains, 0.85, 0.7))
    else:
        loc_clos = np.full(len(m), 0.7)

    # Tipo de propiedad (evitar NA)
    if sel_prop and sel_prop != "— Todos —":
        eq_prop = (m["prop_group"].astype("string") == sel_prop).fillna(False).to_numpy()
        prop_clos = np.where(eq_prop, 1.0, 0.75)
    else:
        prop_clos = np.ones(len(m))

    # Numéricos
    acc_clos  = _range_closeness(m.get("accommodates", pd.Series(np.nan, index=m.index)), *rng_acc)
    bed_clos  = _range_closeness(m.get("bedrooms",     pd.Series(np.nan, index=m.index)), *rng_bed)
    bath_clos = _range_closeness(m.get("bathrooms",    pd.Series(np.nan, index=m.index)), *rng_bath)

    # Rating / reviews
    rating_clos = _rating_closeness(m.get("review_scores_rating", pd.Series(np.nan, index=m.index)), rating_min)
    reviews_clos= _reviews_closeness(m.get("n_reviews",           pd.Series(0,    index=m.index)), n_reviews_min)

    # Amenities
    if sel_amens:
        cols = [c for c in sel_amens if c in m.columns]
        if cols:
            present = m[cols].fillna(0).astype(int).sum(axis=1).to_numpy()
            amen_clos = present / float(len(cols))
        else:
            amen_clos = np.ones(len(m))
    else:
        amen_clos = np.ones(len(m))

    # Score ponderado
    w = {"loc": 3.0, "prop": 1.0, "acc": 1.2, "bed": 1.0, "bath": 1.0, "rating": 1.0, "reviews": 0.6, "amen": 1.0}
    score = (w["loc"]*loc_clos + w["prop"]*prop_clos + w["acc"]*acc_clos +
             w["bed"]*bed_clos + w["bath"]*bath_clos + w["rating"]*rating_clos +
             w["reviews"]*reviews_clos + w["amen"]*amen_clos)

    m = m.assign(_score=score)
    m = m.sort_values("_score", ascending=False, kind="mergesort")
    ids = m["listing_id"].astype(str).head(top_k).tolist()
    return ids

# Mes en español
MESES_ES = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio",
            "Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
def mes_es(d: dt.date) -> str:
    return f"{MESES_ES[d.month-1]} {d.year}"

def month_bounds(month: dt.date):
    start = dt.date(month.year, month.month, 1)
    end = (pd.Timestamp(start) + pd.offsets.MonthEnd(1)).date()
    return start, end

def month_calendar_df(prices: pd.DataFrame, month: dt.date) -> pd.DataFrame:
    if prices.empty:
        return pd.DataFrame(columns=["date","price","dow","week","day"])
    start, end = month_bounds(month)
    idx = pd.date_range(start, end, freq="D", inclusive="both")
    daily = prices.groupby("date", as_index=True)["price"].median()
    s = daily.reindex(idx)
    delta = (pd.Series(idx) - pd.Timestamp(start)).dt.days.to_numpy()
    week = (delta // 7).astype(int)
    df = pd.DataFrame({"date": idx, "day": idx.day, "dow": idx.weekday, "week": week, "price": s.to_numpy()})
    return df

def calendar_chart(cal: pd.DataFrame, month: dt.date):
    import altair as alt
    if cal.empty: return None
    vals = cal["price"].dropna().to_numpy()
    if vals.size==0: return None
    vmin = float(np.percentile(vals, 5))
    vmax = float(np.percentile(vals,95))
    cal = cal.copy()
    cal["dow_name"]  = cal["dow"].map({0:"Lun",1:"Mar",2:"Mié",3:"Jue",4:"Vie",5:"Sáb",6:"Dom"})
    cal["week_str"]  = (cal["week"]+1).astype(str)
    cal["price_str"] = cal["price"].round(0).astype("Int64").astype(str).radd("€")

    title = alt.TitleParams(
        text=mes_es(month), anchor="middle",
        fontSize=26, fontWeight="bold", color="#111827", offset=14
    )
    base = alt.Chart(cal).properties(width="container", height=460, title=title)

    heat = base.mark_rect(radius=8).encode(
        x=alt.X("dow_name:O",
                sort=["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"],
                title="", axis=alt.Axis(orient="top", labelFontSize=12, labelColor="#374151")),
        y=alt.Y("week_str:O", title="", axis=alt.Axis(labelFontSize=12, labelColor="#374151")),
        color=alt.Color("price:Q",
                        scale=alt.Scale(domain=[vmin,vmax], range=["#eef6ff","#dbeafe","#a5b4fc","#60a5fa","#3b82f6"]),
                        legend=None),
        tooltip=[alt.Tooltip("date:T", title="Fecha"),
                 alt.Tooltip("price:Q", title="Precio", format=".0f")]
    )
    text_price = base.mark_text(fontWeight="bold", color="#1f2937", size=12).encode(
        x="dow_name:O", y="week_str:O", text="price_str:N"
    )
    text_day = base.mark_text(align="left", dx=-26, dy=-14, color="#6b7280", size=10).encode(
        x="dow_name:O", y="week_str:O", text="day:Q"
    )
    return (heat + text_day + text_price).interactive()

# -------------- UI --------------
def main():
    # Header + logo
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<h1 class='h1-center'>PRADAR - Price Radar</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-center'>Sistema de recomendación de precios para tu alquiler vacacional (Mallorca)</p>", unsafe_allow_html=True)
    with c1:
        logo = None
        for cand in [ASSETS_DIR/"pradar_logo.png", PROJECT_ROOT/"logo Pradar.png", ASSETS_DIR/"logo Pradar.png"]:
            if cand.exists(): logo = cand; break
        if logo: st.image(str(logo), width=300)

    with st.sidebar:
        meta = load_meta()

        # 📍 Ubicación
        st.markdown("### 📍 Selecciona ubicación")
        loc_opts = build_location_options(meta)
        loc_choice = st.selectbox("Selecciona ubicación", ["— Todos —"] + loc_opts, index=0, key="f_loc")

        # 🧱 Tipo de propiedad
        st.markdown("### 🧱 Tipo de propiedad")
        desired_order = ["Casa/Villa","Apartamento","Hab. privada","Otros"]
        present = [x for x in desired_order if x in meta["prop_group"].dropna().unique().tolist()]
        sel_prop = st.selectbox("Tipo", ["— Todos —"] + present, index=0, key="f_prop")

        # 👥 Nº personas / Dormitorios / Baños
        st.markdown("### 👥 Nº personas / Dormitorios / Baños")
        cA, cB, cC = st.columns(3)

        with cA:
            if "accommodates" in meta.columns and meta["accommodates"].notna().any():
                mn = int(np.nanmin(meta["accommodates"])); mx = int(np.nanmax(meta["accommodates"]))
                mx = min(max(mx, 8), 16)
            else:
                mn, mx = 1, 16
            rng_acc = st.slider("Nº personas", mn, mx, (mn, min(mx,8)), key="f_acc")
            st.markdown(
                f"<div class='smallhelp'>Rango: {rng_acc[0]} — {'16+' if rng_acc[1]>=16 else rng_acc[1]}</div>",
                unsafe_allow_html=True
            )

        with cB:
            if "bedrooms" in meta.columns and meta["bedrooms"].notna().any():
                mn_b = max(0, int(np.nanmin(meta["bedrooms"])))
                mx_b = min(12, int(np.nanmax(meta["bedrooms"])))
            else:
                mn_b, mx_b = 0, 6
            if sel_prop == "Hab. privada":
                mx_b = 1
            rng_bed = st.slider("Dormitorios", mn_b, mx_b, (mn_b, min(mx_b, max(1, mn_b+1))), key="f_bed")

        with cC:
            if "bathrooms" in meta.columns and meta["bathrooms"].notna().any():
                mn_ba = max(0, int(np.nanmin(meta["bathrooms"])))
                mx_ba = min(8, int(np.nanmax(meta["bathrooms"])))
            else:
                mn_ba, mx_ba = 0, 3
            if sel_prop == "Hab. privada":
                mx_ba = 1
            rng_bath = st.slider("Baños", mn_ba, mx_ba, (mn_ba, min(mx_ba, max(1, mn_ba+1))), key="f_bath")

        # ⭐ Review rating y reviews
        st.markdown("### ⭐ Review rating y reviews")
        cR1, cR2 = st.columns(2)
        with cR1:
            vals = pd.to_numeric(meta.get("review_scores_rating"), errors="coerce")
            if np.isfinite(vals).any():
                vals5 = vals/20.0 if (vals.max() and vals.max()>5.1) else vals
                base = float(np.nanmin(vals5))
            else:
                base = 0.0
            rating_min = st.slider("Review rating", 0.0, 5.0, round(base,1), step=0.1, key="f_rating")
        with cR2:
            rev_vals = pd.to_numeric(meta.get("n_reviews"), errors="coerce").fillna(0)
            rev_max  = int(np.nanmax(rev_vals)) if np.isfinite(rev_vals).any() else 0
            max_slider = max(50, rev_max, 1)
            n_reviews_min = st.slider("Nº de reviews (mín.)", 0, max_slider, 0, key="f_nrev")

        # 🧩 Amenities
        st.markdown("### 🧩 Amenities (multi)")
        amen_cols = [c for c in AMENITY_LABELS.keys() if c in meta.columns]
        sel_amens = st.multiselect("Selecciona amenities", options=amen_cols,
                                   format_func=lambda a: AMENITY_LABELS.get(a,a), key="f_amns")

    # --- aplicar filtros estrictos (primer intento) ---
    df = meta.copy()
    if loc_choice != "— Todos —":
        df = df[df["location_display"] == loc_choice]

    if sel_prop != "— Todos —":
        df = df[df["prop_group"]==sel_prop]

    if "accommodates" in df.columns:
        df = df[(df["accommodates"]>=rng_acc[0]) &
                (df["accommodates"]<= (16 if rng_acc[1]>=16 else rng_acc[1]))]

    if "bedrooms" in df.columns:
        df = df[(df["bedrooms"]>=rng_bed[0]) & (df["bedrooms"]<=rng_bed[1])]

    if "bathrooms" in df.columns:
        df = df[(df["bathrooms"]>=rng_bath[0]) & (df["bathrooms"]<=rng_bath[1])]

    if sel_prop == "Hab. privada":
        if "bedrooms" in df.columns:  df = df[df["bedrooms"]<=1]
        if "bathrooms" in df.columns: df = df[df["bathrooms"]<=1]

    if "review_scores_rating" in df.columns:
        r = pd.to_numeric(df["review_scores_rating"], errors="coerce")
        r = r/20.0 if (r.max() and r.max()>5.1) else r
        df = df[r.fillna(0) >= rating_min]

    if "n_reviews" in df.columns:
        df = df[df["n_reviews"] >= n_reviews_min]

    for a in sel_amens:
        if a in df.columns:
            df = df[df[a]==1]

    sel_ids = df["listing_id"].astype(str).unique().tolist() if "listing_id" in df.columns else []

    # --- fuentes de precio ---
    src = load_pricing_source()
    src_path = src["path"]; lid = src["lid"]; dcol = src["dcol"]

    def resolve_price_col(logical):
        colp = PROFILE_TO_COL[logical]
        if colp not in src["all_cols"] and src["name_map"]:
            inv = {v:k for k,v in src["name_map"].items()}
            if colp in inv: colp = inv[colp]
        return colp

    # --- si no hay ids exactos, vecinos más cercanos ---
    if not sel_ids:
        sel_ids = knn_candidate_ids(
            meta=meta,
            loc_choice=loc_choice,
            sel_prop=sel_prop,
            rng_acc=rng_acc,
            rng_bed=rng_bed,
            rng_bath=rng_bath,
            rating_min=rating_min,
            n_reviews_min=n_reviews_min,
            sel_amens=sel_amens,
            top_k=400
        )

    # --- helpers de selección con disponibilidad de precios ---
    def pick_ids_with_prices(profile_col: str, d0: dt.date, d1: dt.date, k:int=200) -> tuple[list[str], pd.DataFrame]:
        df_win = read_prices_window(src_path, lid, dcol, profile_col, d0, d1)
        if df_win.empty:
            return [], df_win
        avail = set(df_win["lid"].astype(str).unique().tolist())
        chosen = [i for i in sel_ids if i in avail][:k]
        return chosen, df_win

    # --- medianas por perfil (siempre con la misma selección) ---
    def profile_median(profile: str) -> float:
        pc = resolve_price_col(profile)
        d0m = dt.date.today(); d1m = d0m + dt.timedelta(days=365-1)
        ids_year, df_year = pick_ids_with_prices(pc, d0m, d1m, k=400)
        if ids_year:
            med = df_year[df_year["lid"].astype(str).isin(ids_year)]["price"].median()
            return float(med) if np.isfinite(med) else 0.0
        med_all = df_year["price"].median() if not df_year.empty else np.nan
        return float(med_all) if np.isfinite(med_all) else 0.0

    medians = {p: profile_median(p) for p in ["Conservador","Normal","Riesgo"]}

    if "profile" not in st.session_state:
        st.session_state.profile = "Normal"  # por defecto

    # --- botones de perfil ---
    bC, bN, bR = st.columns(3, gap="large")
    with bC:
        is_sel = (st.session_state.profile=="Conservador")
        label = f"{('✓ ' if is_sel else '')}{PROFILE_EMOJI['Conservador']} Conservador · €{int(round(medians['Conservador']))}\n{PROFILE_SUB['Conservador']}"
        if st.button(label, key="btn_prof_C", use_container_width=True):
            st.session_state.profile = "Conservador"
        st.markdown(f"<div class='btn-prof btn-C {'btn-selected' if is_sel else ''}'></div>", unsafe_allow_html=True)
    with bN:
        is_sel = (st.session_state.profile=="Normal")
        label = f"{('✓ ' if is_sel else '')}{PROFILE_EMOJI['Normal']} Normal · €{int(round(medians['Normal']))}\n{PROFILE_SUB['Normal']}"
        if st.button(label, key="btn_prof_N", use_container_width=True):
            st.session_state.profile = "Normal"
        st.markdown(f"<div class='btn-prof btn-N {'btn-selected' if is_sel else ''}'></div>", unsafe_allow_html=True)
    with bR:
        is_sel = (st.session_state.profile=="Riesgo")
        label = f"{('✓ ' if is_sel else '')}{PROFILE_EMOJI['Riesgo']} Riesgo · €{int(round(medians['Riesgo']))}\n{PROFILE_SUB['Riesgo']}"
        if st.button(label, key="btn_prof_R", use_container_width=True):
            st.session_state.profile = "Riesgo"
        st.markdown(f"<div class='btn-prof btn-R {'btn-selected' if is_sel else ''}'></div>", unsafe_allow_html=True)

    # --- navegación por meses ---
    st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)
    cprev, cspacer, cnext = st.columns([1,5,1])
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = dt.date.today().replace(day=1)
    with cprev:
        if st.button("◀ Mes anterior", use_container_width=True):
            m = st.session_state.cal_month
            st.session_state.cal_month = (pd.Timestamp(m) - pd.offsets.MonthBegin(1)).date()
    with cspacer:
        st.write("")
    with cnext:
        if st.button("Mes siguiente ▶", use_container_width=True):
            m = st.session_state.cal_month
            st.session_state.cal_month = (pd.Timestamp(m) + pd.offsets.MonthBegin(2)).date().replace(day=1)

    # --- calendario ---
    prof_col = resolve_price_col(st.session_state.profile)
    start_m, end_m = month_bounds(st.session_state.cal_month)
    with st.spinner("Calculando calendario..."):
        ids_month, df_month = pick_ids_with_prices(prof_col, start_m, end_m, k=300)
        if ids_month:
            df_win = df_month[df_month["lid"].astype(str).isin(ids_month)][["date","price"]]
        else:
            d0y = dt.date.today(); d1y = d0y + dt.timedelta(days=365-1)
            ids_year, df_year = pick_ids_with_prices(prof_col, d0y, d1y, k=400)
            if ids_year:
                med = float(df_year[df_year["lid"].astype(str).isin(ids_year)]["price"].median())
            else:
                med = float(df_year["price"].median()) if not df_year.empty else 0.0
            idx = pd.date_range(start_m, end_m, freq="D")
            df_win = pd.DataFrame({"date": idx.date, "price": [med]*len(idx)})
        cal = month_calendar_df(df_win, st.session_state.cal_month)
        chart = calendar_chart(cal, st.session_state.cal_month)

    if chart is not None:
        st.altair_chart(chart, use_container_width=True)

    # --- descarga 365 días (mediana diaria) ---
    d0 = dt.date.today(); d1 = d0 + dt.timedelta(days=365-1)
    ids_year, df_year = pick_ids_with_prices(prof_col, d0, d1, k=400)
    if ids_year:
        df_long = df_year[df_year["lid"].astype(str).isin(ids_year)][["date","price"]]
        daily = (df_long.groupby("date", as_index=False)["price"].median()
                 .rename(columns={"price": f"precio_{st.session_state.profile.lower()}" }))
    else:
        med_all = float(df_year["price"].median()) if not df_year.empty else 0.0
        idx = pd.date_range(d0, d1, freq="D")
        daily = pd.DataFrame({
            "date": idx.date,
            f"precio_{st.session_state.profile.lower()}": [med_all]*len(idx)
        })

    st.download_button(
        "💾 Descargar CSV 365 días (mediana diaria)",
        daily.to_csv(index=False).encode("utf-8"),
        file_name=f"prices_{st.session_state.profile.lower()}_{d0.isoformat()}_{d1.isoformat()}.csv",
        mime="text/csv",
        use_container_width=True
    )

    st.markdown(
        "<div style='text-align:right;color:#9aa3b2;font-size:12px;'>PRADAR • Price Radar</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
