# Análisis multitemporal de patrones de generación de drenaje minero en botaderos de estéril.

Este repositorio contiene los códigos usados en la metodología de mi tesis *Análisis multitemporal de patrones de generación de drenaje minero en botaderos de estéril* para priorizar zonas en botaderos con mayor probabilidad de oxidación de sulfuros, integrando índices multiespectrales, temperatura superficial y litología.

## Qué hay en este repo
- `botaderosdetec.py`: Pipeline en Python + Google Earth Engine para:
  - entrenar un Random Forest con polígonos etiquetados,
  - clasificar escenas,
  - calcular áreas por clase,
  - exportar visualizaciones (PNG).
- `bandas3.py`: Variante enfocada en OBIA (SNIC) + texturas GLCM** y búsqueda aleatoria de subconjuntos de bandas, guardando resultados en CSV.


## Requisitos
- Python 3.9+ 
- Cuenta en **Google Earth Engine** (GEE)
- Librerias:
  - `earthengine-api`
  - `requests`
  - `selenium`
  - `webdriver-manager`
