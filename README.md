# Análisis multitemporal de patrones de generación de drenaje minero en botaderos de estéril.

Este repositorio contiene los códigos usados en la metodología de mi tesis *Análisis multitemporal de patrones de generación de drenaje minero en botaderos de estéril* para priorizar zonas en botaderos con mayor probabilidad de oxidación de sulfuros, integrando índices multiespectrales, temperatura superficial y litología.

## Qué hay en este repo:

### Scripts (.py)
- `botaderosdetec.py`  
  Codigo en Python + Google Earth Engine para clasificación supervisada con Random Forest usando polígonos etiquetados.  
  Incluye: creación de composites multitemporales , cálculo de índices espectrales/derivados, *nmascarado (p. ej., agua), clasificación y export de PNGs (descarga por `getThumbURL`, con fallback vía Selenium).

- `bandas3.py`  
  Codigo en Python + Google Earth Engine: construye un stack grande de features (índices espectrales, compuestos térmicos, métricas/morfología, texturas GLCM, etc.), segmenta con SNIC, calcula estadísticas por objeto, y entrena/valida Random Forest con un random search de subconjuntos de bandas para encontrar combinaciones que maximizan métricas (accuracy/kappa), guardando resultados en CSV.

### Notebooks (Jupyter)

- `Extraccion_de_datos-checkpoint.ipynb`  
  Notebook para extracción de variables por punto y por fecha desde varias fuentes:
  - **Cambios Sentinel-1**: calcula métricas de cambio multitemporal por punto de malla y exporta un CSV resumen (y mapas HTML de apoyo).
  - **Bandas/índices por sensor**: extrae valores para Landsat 8/9 y **Sentinel-2 por `punto_id` y exporta un CSV por sensor.
  - **DEM**: muestrea elevación por punto desde **NASADEM** y **Copernicus GLO30**, exportando CSVs.
  - **Clima**: extrae variables diarias desde **ERA5-LAND** y los guarda en un excel.

- `Datos_Litologia-checkpoint.ipynb`  
  Implementa SAM (Spectral Angle Mapper) en GEE con un catálogo de minerales/firma espectral, extraido de la USGS. Clasifica escenas de Sentinel-2, aplica umbrales de confianza, remapea a clases de litología (p. ej., Skarn/Intrusiva/Caliza/Indeterminada), hace *sampling* en la malla/puntos (`numero`, `X`, `Y`) y exporta CSV consolidado con `fecha_img`, `litologia`, `mineral'.

- `Merge_CSV-checkpoint.ipynb`  
  Notebook de integración final: une los CSV de bandas (S2/L8/L9) en un solo dataset normalizando `punto_id` y `fecha`, luego integra el CSV de cambios Sentinel-1, agrega clima (Excel ERA5-LAND), une coordenadas + elevación (DEM) y finalmente incorpora litología/mineral (SAM). 


## Requisitos
- Python 3.9+ 
- Cuenta en Google Earth Engine (GEE)
- Librerias:
  - `earthengine-api`
  - `requests`
  - `selenium`
  - `webdriver-manager`
