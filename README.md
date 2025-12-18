# Análisis multitemporal de patrones de generación de drenaje minero en botaderos de estéril (sensores remotos)

Este repositorio contiene los **códigos** usados en la metodología para priorizar zonas en botaderos con mayor probabilidad de oxidación de sulfuros, integrando índices multiespectrales, temperatura superficial y litología.

## Qué hay en este repo
- `botaderosdetec.py`: Pipeline en **Python + Google Earth Engine** para:
  - entrenar un **Random Forest** con polígonos etiquetados,
  - clasificar escenas,
  - calcular áreas por clase,
  - exportar visualizaciones (PNG).
- `bandas3.py`: Variante enfocada en **OBIA (SNIC) + texturas GLCM** y **búsqueda aleatoria de subconjuntos de bandas**, guardando resultados en CSV.

## Metodología
- Variables espectrales y térmicas: Fe³⁺/Fe²⁺, humedad (NMDI) y temperatura superficial (LST).
- Litología: clasificación con **SAM** usando firmas espectrales (Sentinel-2).
- Cambios: detección multitemporal con **Sentinel-1 (SAR)**.
- Clasificación: enfoques supervisados en **Google Earth Engine** + opción orientada a objetos (**SNIC/OBIA**) + **texturas GLCM**.
- Priorización: protocolo por “kernels/ventanas” para detectar patrones persistentes (Fe³⁺ alto + NMDI bajo + tendencias Fe²⁺↓ y Fe³⁺↑).

## Requisitos
- Python 3.9+ 
- Cuenta habilitada en **Google Earth Engine** (GEE)
- Paquetes:
  - `earthengine-api`
  - `requests`
  - `selenium`
  - `webdriver-manager`
