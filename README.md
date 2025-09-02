# PRADAR · Calculadora de Precios VR (App Streamlit)

Aplicación **read‑only** para **visualizar y descargar precios diarios** generados por el pipeline PRADAR (notebooks `00→04`).  
> **No re‑entrena** modelos ni **optimiza** precios: **consume artefactos** exportados por los cuadernos.

- **Público objetivo**: pequeños propietarios (1–3 listings).
- **Diferenciación**: simplicidad, **hiperlocalidad**, bajo coste.
- **Perfiles**: **Conservador / Normal / Riesgo** (C/N/R) — se aceptan sinónimos “Agresivo” (A) ⇄ “Riesgo” (R).

---

## 1) Estructura esperada del proyecto

```text
PRADAR/
  app/
    app.py
    README.md
    requirements.txt
    images/
      logo_pradar_main.png        # opcional
  data/
    processed/
      df_prices.parquet           # o df_prices.csv
      df_prices_eta.parquet       # opcional
      app_prices_active.parquet   # opcional (preferente si existe)
      eta.json                    # opcional (elasticidad operativa)
  models/                         # artefactos de notebooks (solo lectura)
  reports/                        # calibración, backtests, etc. (solo lectura)
```

La app **detecta automáticamente** la raíz del proyecto (`PRADAR_ROOT`) buscando `data/processed` hacia arriba desde `app/`.  
También puedes fijarla mediante variable de entorno:

```bash
# Linux/macOS
export PRADAR_ROOT="/ruta/a/PRADAR"

# Windows PowerShell
$env:PRADAR_ROOT="C:\ruta\a\PRADAR"
```

---

## 2) Entrada de datos (Contrato)

**Orden de preferencia de ficheros** que la app intentará cargar en `"$PRADAR_ROOT/data/processed/"`:

1. `app_prices_active.parquet` (o `.csv`)
2. `df_prices_eta.parquet`
3. `df_prices.parquet`
4. `price_recomendado.parquet` / `price_recomendado.csv`

> Basta con **uno** de los anteriores. Parquet es preferible por rendimiento.

### 2.1 Columnas **obligatorias** (cualquiera de estas variantes)

**Variante A (recomendada, con perfiles):**
- `date` (YYYY‑MM‑DD)
- **Precios por perfil** (se admiten sinónimos):
  - `price_conservador`, `price_normal`, `price_riesgo`
  - **o** `price_C`, `price_N`, `price_R`
  - **opcional**: `price_agresivo` ⇄ `price_A` (si tu política usa A en lugar de R)

**Variante B (fallback mínimo):**
- `date`, `price_optimal`

### 2.2 Columnas **opcionales** (habilitan filtros y tooltips adicionales)

- Identidad/geo: `listing_id`, `city`, `municipality`, `neighbourhood`
- Producto: `capacity`, `minimum_nights`, `maximum_nights`, `amenities`
- Calidad: `rating`, `instant_bookable`, `superhost`
- Señales de control: `p_cal`, `alpha_star`, `lead_time`, `guardrail_hit`, `cap_high`, `cap_low`

**Tipos esperados**
- `date`: fecha ISO o string ISO (la app la convierte a fecha).
- precios: numéricos (float).
- booleanos: `instant_bookable`, `superhost`, `guardrail_hit`.
- listas/strings: `amenities` admite lista serializada como string separada por “;”.

---

## 3) Instalación

Requisitos: **Python 3.10–3.12** (recomendado 3.11) y `pip` actualizado.

```bash
# Linux/macOS
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Windows PowerShell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

---

## 4) Ejecución

Desde la carpeta `app/` (o donde esté `app.py`):

```bash
streamlit run app.py
```

**Flujo de uso**
1. Selecciona el **fichero** detectado (si hay más de uno).
2. Elige **perfil** (C/N/R o A) y **rango de fechas** (por defecto 365 días).
3. Aplica **filtros** (ubicación, capacidad, estancia mín/máx, rating, instant booking, superhost) si hay columnas.
4. Explora el **calendario** (base vs recomendado, dif. vs Normal) y descarga **CSV** mensual.

---

## 5) Detalles funcionales

- **Calendario 365** con **tooltips** (si existen columnas): `p_cal`, `lead_time`, `guardrail_hit`.
- **Perfiles**: Conservador / Normal / Riesgo (o Agresivo). La app mapea sinónimos automáticamente.
- **Export**: CSV mensual con las columnas visibles en la UI (perfil seleccionado).
- **Robustez**: si faltan columnas opcionales, la UI oculta filtros asociados (fallo suave).

---

## 6) Resolución de problemas

**“No se encuentra fichero de precios”**
- Asegúrate de que existe al menos uno en `"$PRADAR_ROOT/data/processed/"` y que `PRADAR_ROOT` apunta a la raíz correcta.

**“Schema inválido / faltan columnas”**
- Cumple **Variante A** o **B** de columnas mínimas y tipos. Comprueba nulos y formatos de fecha.

**Error `ArrowInvalid`/`pyarrow` o lectura Parquet**
- Usa las versiones de `pyarrow`/`pandas` del `requirements.txt`. Evita Parquet escritos con motores muy antiguos.

**Rendimiento bajo al filtrar**
- Prefiere Parquet a CSV; reduce el rango de fechas o filtra un subconjunto de listings.

---

## 7) Privacidad, alcance y disclaimer

La app trabaja en modo **read‑only** con artefactos anonimizados. **No** gestiona datos personales de huéspedes/anfitriones.

> **Disclaimer**: PRADAR ofrece **estimaciones** y **recomendaciones** basadas en datos históricos; **no garantiza** resultados futuros.

---

## 8) Créditos y alineación con el pipeline

- **Notebooks**: `00_orchestration` (ETL) → `01_eda` → `02_feature_engineering` → `03_modeling` (XGBoost + calibración isotónica) → `04_price_optimization` (grid + guardrails).
- **Artefactos** (solo lectura): `preproc.pkl`, `xgb_pradar.json`, `calib_isotonic.pkl`, `feature_names.json`.
- **Salidas de pricing**: `df_prices.parquet/csv` (o `df_prices_eta.parquet`, `app_prices_active.parquet`) con `price_C/N/R` (o `price_conservador/normal/riesgo`) y, si procede, `price_A`/`price_agresivo`.
- **Rutas**: por defecto `"$PRADAR_ROOT/data/processed/"`.
