from __future__ import annotations
import os
import time
from typing import List, Tuple
import ee
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

ee.Initialize()

# 0. PARÁMETROS

_PALETTE = [
    "#FF0000",  # Botadero (rojo)
    "#FFFF00",  # Relave   (amarillo)
    "#00FF00",  # Rajo     (verde)
    "#0000FF",  # Agua     (azul)
    "#8000FF"   # Área mina (morado)
]
_VIS_RGB = {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0.03, "max": 0.35, "gamma": 1.5}

RF_PARAMS = {
    "numberOfTrees": 700,
    "bagFraction": 0.8,
    "minLeafPopulation": 3,
    "maxNodes": 50,
    "seed": 42
}

SAMPLE_PARAMS = {
    "max_train": 8000,
    "max_test": 2000,
    "split": 0.7
}

TRAIN_IDS = [
    "LC08_008067_20170618",
    "LC08_008067_20170720",
    "LC08_008067_20230705",
    "LC08_008067_20230518",
    "LC08_008067_20140101",
    "LC08_008067_20140101"
]

# 1. ETIQUETADO DE POLÍGONOS

def _add_label(feat: ee.Feature) -> ee.Feature:
    name = ee.String(
        ee.Algorithms.If(feat.propertyNames().contains("Name"), feat.get("Name"), "")
    ).toLowerCase()
    etiqueta = ee.Number(
        ee.Algorithms.If(name.index("botadero").gte(0), 1,
        ee.Algorithms.If(name.index("relave").gte(0),   2,
        ee.Algorithms.If(name.index("rajo").gte(0),     3,
        ee.Algorithms.If(name.index("agua").gte(0),     4,
        ee.Algorithms.If(name.index("mina").gte(0),     5,
        0)))))
    )
    return feat.set("etiqueta", etiqueta)

# 2. COMPOSITE MULTITEMPORAL 

def get_composite(img_id: str, buffer_days: int = 3, maxCloud: int = 30, roi: ee.Geometry = None) -> ee.Image:
    """
    Crea un composite median de ±buffer_days autour de img_id para reducir nubes.
    """
    # Extraer fecha de img_id
    img = ee.Image(f"LANDSAT/LC08/C02/T1_L2/{img_id}")
    date = ee.Date(img.get('system:time_start'))
    start = date.advance(-buffer_days, 'day')
    end   = date.advance(buffer_days,  'day')
    col = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start, end)
          .filterBounds(roi or ee.Geometry.Point(0,0))
          .filter(ee.Filter.lt('CLOUD_COVER', maxCloud))
          .map(apply_scale_factors_l8)
          .map(_add_indices)
    )
    return col.median().set('system:time_start', date.millis())

# 2. BANDAS E ÍNDICES
    
def _add_indices(img: ee.Image) -> ee.Image:
    # Bandas base
    blue, green, red = img.select('SR_B2'), img.select('SR_B3'), img.select('SR_B4')
    nir, swir1, swir2 = img.select('SR_B5'), img.select('SR_B6'), img.select('SR_B7')
    tir = img.select('ST_B10')
    eps = ee.Number(1e-6)
    dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation').clip(img.geometry()).resample('bilinear').reproject(crs=img.projection(), scale=img.projection().nominalScale())
    elev = dem.rename('elevation')
    slope = ee.Terrain.slope(dem).rename('slope')
    mean9 = dem.focal_mean(9, 'circle', 'pixels')
    tpi = dem.subtract(mean9).rename('TPI')
    rug = dem.focal_max(3, 'circle', 'pixels').subtract(dem.focal_min(3, 'circle', 'pixels')).rename('Rugosity')
    curva = dem.convolve(ee.Kernel.laplacian8()).rename('Curvature')
    ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    savi = img.expression('((NIR-RED)*(1+L))/(NIR+RED+L)', {'NIR': nir, 'RED': red, 'L': 0.5}).rename('SAVI')
    evi = img.expression('2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)', {'NIR': nir, 'RED': red, 'BLUE': blue}).rename('EVI')
    ndwi = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    vari   = green.subtract(red).divide(green.add(red).subtract(blue)).rename('VARI')
    mndwi = img.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    ndmi = img.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI')
    wdi = swir2.subtract(nir).divide(swir2.add(nir).add(eps)).rename('WDI')
    bsi = img.expression('((RED+SWIR1)-(NIR+BLUE))/((RED+SWIR1)+(NIR+BLUE)+eps)', {'RED': red, 'SWIR1': swir1, 'NIR': nir, 'BLUE': blue, 'eps': eps}).rename('BSI')
    ndbi = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    nbli = img.normalizedDifference(['SR_B4', 'ST_B10']).rename('NBLI')
    tndi = img.normalizedDifference(['ST_B10', 'SR_B5']).rename('TNDI')
    tnr = tir.divide(nir.add(eps)).rename('TNR')
    clay_ratio = swir1.divide(swir2.add(eps)).rename('Clay_Ratio')
    fe_oxide = red.divide(blue.add(eps)).rename('Fe_Oxide_Index')
    fe_comp = red.add(swir1).divide(nir.add(eps)).rename('Fe_Composite_Index')
    clay = swir1.divide(swir2.add(eps)).rename('Clay_Index')
    tcb = img.expression('0.3029*B+0.2786*G+0.4733*R+0.5599*N+0.508*S1+0.1872*S2', {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}).rename('TCB')
    tcg = img.expression('-0.2941*B-0.243*G-0.5424*R+0.7276*N+0.0713*S1-0.1608*S2', {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}).rename('TCG')
    tcw = img.expression('0.1509*B+0.1973*G+0.3279*R-0.7112*N-0.4572*S1+0.6636*S2', {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}).rename('TCW')
    tci = img.expression('0.1511*B2 + 0.1973*B3 + 0.3283*B4 + 0.3407*B5 - 0.7117*B6 - 0.4559*B7', {'B2': blue, 'B3': green, 'B4': red, 'B5': nir, 'B6': swir1, 'B7': swir2}).rename('TCI')
    b5_u = nir.multiply(100).toUint8()
    b6_u = swir1.multiply(100).toUint8()
    b7_u = swir2.multiply(100).toUint8()
    g5_3 = b5_u.glcmTexture(3)
    g5_7 = b5_u.glcmTexture(7)
    g6_3 = b6_u.glcmTexture(3)
    g6_7 = b6_u.glcmTexture(7)
    g7_3 = b7_u.glcmTexture(3)
    glcm_SR_B5_con_3 = g5_3.select('SR_B5_contrast').rename('GLCM_SR_B5_contrast_3')
    glcm_SR_B5_var_3 = g5_3.select('SR_B5_var').rename('GLCM_SR_B5_var_3')
    glcm_SR_B5_con_7 = g5_7.select('SR_B5_contrast').rename('GLCM_SR_B5_contrast_7')
    glcm_SR_B6_con_3 = g6_3.select('SR_B6_contrast').rename('GLCM_SR_B6_contrast_3')
    glcm_SR_B6_idm_3 = g6_3.select('SR_B6_idm').rename('GLCM_SR_B6_idm_3')
    glcm_SR_B6_ent_3 = g6_3.select('SR_B6_ent').rename('GLCM_SR_B6_ent_3')
    glcm_SR_B6_var_3 = g6_3.select('SR_B6_var').rename('GLCM_SR_B6_var_3')
    glcm_SR_B6_con_7 = g6_7.select('SR_B6_contrast').rename('GLCM_SR_B6_contrast_7')
    glcm_SR_B6_idm_7 = g6_7.select('SR_B6_idm').rename('GLCM_SR_B6_idm_7')
    glcm_SR_B6_ent_7 = g6_7.select('SR_B6_ent').rename('GLCM_SR_B6_ent_7')
    glcm_SR_B7_con_3 = g7_3.select('SR_B7_contrast').rename('GLCM_SR_B7_contrast_3')
    # Morfología granulométrica
    open3 = swir1.focal_min(3, 'circle', 'pixels').focal_max(3, 'circle', 'pixels').rename('Open_3')
    close3 = swir1.focal_max(3, 'circle', 'pixels').focal_min(3, 'circle', 'pixels').rename('Close_3')
    gran7 = swir1.focal_max(7, 'circle', 'pixels').subtract(swir1.focal_min(7, 'circle', 'pixels')).rename('Granulo_7')

    return img.addBands([
        ndvi, savi, evi, ndwi, mndwi, ndmi, wdi, bsi, ndbi, nbli, tndi, tnr,
        clay_ratio, fe_oxide, fe_comp, clay,
        tcb, tcg, tcw, tci,
        elev, slope, tpi, rug, curva,
        glcm_SR_B5_con_3, glcm_SR_B5_var_3, glcm_SR_B5_con_7,
        glcm_SR_B6_con_3, glcm_SR_B6_idm_3, glcm_SR_B6_ent_3, glcm_SR_B6_var_3,
        glcm_SR_B6_con_7, glcm_SR_B6_idm_7, glcm_SR_B6_ent_7,
        glcm_SR_B7_con_3,
        open3, close3, gran7,vari
    ])

# 3. POST-PROCESADO JERÁRQUICO

def mask_water(img: ee.Image, thr: float = 0.1) -> ee.Image:
    """Máscara simple de agua usando NDWI."""
    return img.select('NDWI').gte(thr)

def hierarchical_classify(
    img: ee.Image,
    clf: ee.Classifier,
    bands: List[str],
    water_thr: float = 0.1
) -> ee.Image:

    water_mask = mask_water(img, water_thr)
    water_cls  = water_mask.multiply(4).rename('classification')
    non_water  = img.updateMask(water_mask.Not())
    rf_cls     = non_water.select(bands).classify(clf).rename('classification')
    return rf_cls.unmask(water_cls)

# 4. CARGA Y ESCALA

def apply_scale_factors_l8(image: ee.Image) -> ee.Image:
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

def _load_landsat_single(img_id: str) -> ee.Image:
    img = ee.Image(f"LANDSAT/LC08/C02/T1_L2/{img_id}")
    img = apply_scale_factors_l8(img)
    return _add_indices(img)


# 5. ENTRENAMIENTO RF

def train_rf(
    polygons: ee.FeatureCollection,
    img_ids: List[str],
    scale: int = 30
)   Tuple[ee.Classifier, List[str]]:
    bands =[
    'NDVI', 'SAVI', 'EVI', 'NDWI', 'MNDWI', 'NDMI', 'WDI', 'BSI', 'NDBI',
    'NBLI', 'TNDI', 'TNR',
    'elevation',
    'GLCM_SR_B5_contrast_7', 'GLCM_SR_B6_idm_7', 'GLCM_SR_B6_ent_7',
    'Open_3', 'Close_3',
    'TCI',
    'Clay_Ratio','VARI'
]

    polygons = polygons.map(lambda f: f)
    samples = ee.FeatureCollection([])
    classes = polygons.aggregate_array('etiqueta').distinct().getInfo()
    per_class = SAMPLE_PARAMS['max_train'] // len(classes)
    for tid in img_ids:
        img = get_composite(tid, roi=polygons.geometry())
        for c in classes:
            fc = img.select(bands).sampleRegions(
                collection=polygons.filter(ee.Filter.eq('etiqueta', c)),
                properties=['etiqueta'],
                scale=scale
            ).randomColumn('rnd', RF_PARAMS['seed']).limit(per_class)
            samples = samples.merge(fc)
    samples = samples.randomColumn('rnd', RF_PARAMS['seed'])
    train = samples.filter(ee.Filter.lt('rnd', SAMPLE_PARAMS['split']))
    test  = samples.filter(ee.Filter.gte('rnd', SAMPLE_PARAMS['split']))
    clf = ee.Classifier.smileRandomForest(**RF_PARAMS).train(
        features=train, classProperty='etiqueta', inputProperties=bands
    )
    matrix = test.classify(clf).errorMatrix('etiqueta','classification')
    print('Accuracy:', matrix.accuracy().multiply(100).getInfo(), '%')
    print('Kappa:',   matrix.kappa().getInfo())
    return clf, bands

# 6. CÁLCULO DE ÁREAS

def calcular_areas_por_clase(clasificada: ee.Image, roi: ee.Geometry) -> dict:
    groups = ee.Image.pixelArea().addBands(clasificada) \
        .reduceRegion(
            reducer=ee.Reducer.sum().group(1,'clase'),
            geometry=roi,
            scale=30,
            maxPixels=1e13
        ).get('groups')
    return ee.List(groups).getInfo()

# 7. EXPORTAR PNG

def export_png(
    blended: ee.Image,
    img_id: str,
    roi: ee.Geometry,
    out_dir: str,
    prefix: str = 'rf',
    scale: int = 30
):
    coords = roi.bounds().getInfo()['coordinates'][0]
    xs, ys = zip(*coords)
    region = [min(xs), min(ys), max(xs), max(ys)]
    url = blended.getThumbURL({"region": region, "scale": scale, "format": "png", "transparent": True})
    outfile = os.path.join(out_dir, f"{img_id}_{prefix}.png")
    try:
        with requests.get(url, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(outfile, 'wb') as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        print(f"PNG saved: {outfile}")
    except ee.EEException:
        options = Options(); options.add_argument('--headless'); options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.set_window_size(800, 800)
        driver.get(url)
        img_el = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.TAG_NAME, 'img')))
        with open(outfile, 'wb') as f:
            f.write(img_el.screenshot_as_png)
        driver.quit()
        print(f"PNG saved via Selenium: {outfile}")

def main():
    out_dir = r"C:\Users\Natascha\Desktop\Tesis\pantallazos"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1) Cargar y etiquetar polígonos
    fc2023 = ee.FeatureCollection(
        "projects/ee-tu_proyecto/assets/2023").map(_add_label)
    fc2017 = ee.FeatureCollection(
        "projects/ee-tu_proyecto/assets/2017").map(_add_label)
    fc2014 = ee.FeatureCollection(
        "projects/ee-tu_proyecto/assets/20144").map(_add_label)

    polygons = (fc2014.merge(fc2023).merge(fc2017).filter(ee.Filter.gt('etiqueta', 0)))
    print('Class counts:', polygons.aggregate_histogram('etiqueta').getInfo())
    area_mina_fc = polygons.filter(ee.Filter.eq('etiqueta', 5))
    roi = area_mina_fc.geometry()
    clf, bands = train_rf(polygons, TRAIN_IDS)

    for tid in TRAIN_IDS:
        start = time.time()
        img = get_composite(tid, roi=roi)
        # Clasificación jerárquica y limpieza morfológica
        classified = hierarchical_classify(img, clf, bands)
        cleaned = (classified
            .focal_mode(kernel=ee.Kernel.square(1))
            .focal_min(kernel=ee.Kernel.square(1))
            .focal_max(kernel=ee.Kernel.square(1)))

        val_fc = img.select(bands).addBands(cleaned.rename('classification')).sampleRegions(
            collection=polygons.filterBounds(roi),
            properties=['etiqueta'],
            scale=30
        )
        post_matrix = val_fc.errorMatrix('etiqueta', 'classification')
        print(f"Post-classification Confusion Matrix for {tid}: {post_matrix.getInfo()}")
        print(f"Post-classification Accuracy for {tid}: {post_matrix.accuracy().multiply(100).getInfo()} %")
        print(f"Post-classification Kappa for {tid}: {post_matrix.kappa().getInfo()}")

        areas = calcular_areas_por_clase(cleaned, roi)
        print(f"Areas {tid}:")
        for g in areas:
            print(f"  Class {g['clase']}: {round(g['sum']):,} m²")

        rgb = img.visualize(**_VIS_RGB).clip(roi)
        vis_rf = cleaned.visualize(min=1, max=5, palette=_PALETTE, opacity=0.4).clip(roi)
        blended = rgb.blend(vis_rf)
        export_png(blended, tid, roi, out_dir, prefix="RF")
        print(f"{tid} done in {time.time()-start:.1f}s")


if __name__ == '__main__':
    main()
