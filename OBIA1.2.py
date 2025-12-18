
from __future__ import annotations
import os
import time
from typing import List, Tuple
from requests.exceptions import RequestException
import ee
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from ee import EEException

ee.Initialize()

# --------------------------------------------------------------
# 0. PARÁMETROS
# --------------------------------------------------------------
_PALETTE = [
    "#FF0000",  # Botadero (rojo)
    "#FFFF00",  # Relave   (amarillo)
    "#00FF00",  # Rajo     (verde)
    "#0000FF",  # Agua     (azul)
    "#8000FF"   # Área mina (morado)
]
_VIS_RGB = {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0.03, "max": 0.35, "gamma": 1.5}

RF_PARAMS = {
    "numberOfTrees": 200,
    "bagFraction": 0.8,
    "minLeafPopulation": 3,
    "maxNodes": 50,
    "seed": 42
}

SAMPLE_PARAMS = {
    "max_train": 8000,
    "max_test": 500,
    "split": 0.7
}


TRAIN_IDS = [
    "LC08_008067_20170618",
    "LC08_008067_20170720",
    "LC08_008067_20230705",
    "LC08_008067_20230518",
    "LC08_008066_20141219",
]


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

def get_composite(img_id: str, buffer_days: int = 3, maxCloud: int = 30, roi: ee.Geometry = None) -> ee.Image:
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

def compute_metrics(matrix, retries=5, wait=20):

    # Creamos un diccionario con las dos métricas
    metrics = ee.Dictionary({
        'acc':   matrix.accuracy().multiply(100),
        'kappa': matrix.kappa()
    })
    for i in range(retries):
        try:
            info = safe_get_info(metrics)
            return info['acc'], info['kappa']
        except EEException as e:
            if "Too many concurrent aggregations" in str(e):
                print(f"Warning: intento {i+1} fallido (429). Reintentando en {wait}s…")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("No se pudieron calcular métricas tras varios reintentos.")

def _add_indices(img: ee.Image) -> ee.Image:
    # Bandas base
    blue, green, red = img.select('SR_B2'), img.select('SR_B3'), img.select('SR_B4')
    nir, swir1, swir2 = img.select('SR_B5'), img.select('SR_B6'), img.select('SR_B7')
    tir = img.select('ST_B10')
    eps = ee.Number(1e-6)

    # Índices espectrales
    ndvi    = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    savi    = img.expression('((NIR-RED)*(1+L))/(NIR+RED+L)', {'NIR': nir, 'RED': red, 'L': 0.5}).rename('SAVI')
    evi     = img.expression('2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)', {'NIR': nir, 'RED': red, 'BLUE': blue}).rename('EVI')
    evi2    = img.expression('2.4*(NIR-RED)/(NIR+RED+1)', {'NIR': nir, 'RED': red}).rename('EVI2')
    msavi   = img.expression('(2*NIR + 1 - sqrt((2*NIR + 1)**2 - 8*(NIR - RED))) / 2', {'NIR': nir, 'RED': red}).rename('MSAVI')
    osavi   = img.expression('((NIR-RED)*(1+0.16))/(NIR+RED+0.16)', {'NIR': nir, 'RED': red}).rename('OSAVI')
    ndwi    = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    mndwi   = img.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    ndmi    = img.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI')
    ndbi    = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    nbr     = img.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')
    nbr2    = img.normalizedDifference(['SR_B6', 'SR_B7']).rename('NBR2')
    psri    = img.expression('(RED - BLUE) / (GREEN + eps)', {'RED': red, 'BLUE': blue, 'GREEN': green, 'eps': eps}).rename('PSRI')
    gci     = img.normalizedDifference(['SR_B5', 'SR_B3']).rename('GCI')
    vari    = img.expression('(GREEN - RED)/(GREEN + RED - BLUE)', {'GREEN': green, 'RED': red, 'BLUE': blue}).rename('VARI')
    vnsir   = red.add(green).add(blue).divide(swir1.add(swir2).add(eps)).rename('VNSIR')
    ndsi    = img.normalizedDifference(['SR_B6', 'SR_B2']).rename('NDSI')
    wdi     = swir2.subtract(nir).divide(swir2.add(nir).add(eps)).rename('WDI')
    bsi     = img.expression('((RED+SWIR1)-(NIR+BLUE))/((RED+SWIR1)+(NIR+BLUE)+eps)',
               {'RED': red, 'SWIR1': swir1, 'NIR': nir, 'BLUE': blue, 'eps': eps}).rename('BSI')
    gndvi   = img.normalizedDifference(['SR_B5', 'SR_B3']).rename('GNDVI')
    bai     = img.expression('1 / ((0.1 - RED)**2 + (0.06 - NIR)**2)',
               {'RED': red, 'NIR': nir}).rename('BAI')
    emi     = swir1.subtract(nir).divide(swir1.add(nir).add(eps)).rename('EMI')
    ri      = red.divide(green.add(eps)).rename('RI')

    # Geoquímicos
    fmr         = swir1.divide(nir.add(eps)).rename('FMR')
    clay_ratio = swir1.divide(swir2.add(eps)).rename('Clay_Ratio')
    fe_oxide    = red.divide(blue.add(eps)).rename('Fe_Oxide_Index')
    fe_comp     = red.add(swir1).divide(nir.add(eps)).rename('Fe_Composite_Index')
    clay        = swir1.divide(swir2.add(eps)).rename('Clay_Index')
    cri1        = blue.divide(green.add(eps)).rename('CRI1')
    cri2        = red.divide(blue.add(eps)).rename('CRI2')
    mbsi        = img.expression('(SWIR1 - GREEN)/(SWIR1 + GREEN + eps)', {'SWIR1': swir1, 'GREEN': green, 'eps': eps}).rename('MBSI')

    # Derivados térmicos
    nbli    = img.normalizedDifference(['SR_B4', 'ST_B10']).rename('NBLI')
    tndi    = img.normalizedDifference(['ST_B10', 'SR_B5']).rename('TNDI')
    tnr     = tir.divide(nir.add(eps)).rename('TNR')
    bt      = tir.multiply(0.055).add(149).rename('BT')
    shadow_index = blue.add(green).divide(nir.add(eps)).rename('Shadow_Index')
    si_therm     = shadow_index.multiply(tir).rename('SI_Thermal')
    ari          = nir.divide(green.add(eps)).rename('ARI')

    # Tasselled Cap
    tcb = img.expression(
        '0.3029*B + 0.2786*G + 0.4733*R + 0.5599*N + 0.508*S1 + 0.1872*S2',
        {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}
    ).rename('TCB')
    tcg = img.expression(
        '-0.2941*B - 0.243*G - 0.5424*R + 0.7276*N + 0.0713*S1 - 0.1608*S2',
        {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}
    ).rename('TCG')
    tcw = img.expression(
        '0.1509*B + 0.1973*G + 0.3279*R - 0.7112*N - 0.4572*S1 + 0.6636*S2',
        {'B': blue, 'G': green, 'R': red, 'N': nir, 'S1': swir1, 'S2': swir2}
    ).rename('TCW')
    tci = img.expression(
        '0.1511*B2 + 0.1973*B3 + 0.3283*B4 + 0.3407*B5 - 0.7117*B6 - 0.4559*B7',
        {'B2': blue, 'B3': green, 'B4': red, 'B5': nir, 'B6': swir1, 'B7': swir2}
    ).rename('TCI')

    # Texturas GLCM
    def glcm_stats(band_u, name):
        stats = []
        for w in [3, 5, 7]:
            g = band_u.glcmTexture(w)
            stats += [
                g.select(f'{name}_contrast').rename(f'GLCM_{name}_contrast_{w}'),
                g.select(f'{name}_var').rename     (f'GLCM_{name}_var_{w}'),
                g.select(f'{name}_idm').rename     (f'GLCM_{name}_idm_{w}'),
                g.select(f'{name}_ent').rename     (f'GLCM_{name}_entropy_{w}'),
                g.select(f'{name}_diss').rename    (f'GLCM_{name}_dissimilarity_{w}')
            ]
        return stats

    b5_u = nir.multiply(100).toUint8()
    b6_u = swir1.multiply(100).toUint8()
    b7_u = swir2.multiply(100).toUint8()
    glcm_SR_B5 = glcm_stats(b5_u, 'SR_B5')
    glcm_SR_B6 = glcm_stats(b6_u, 'SR_B6')
    glcm_SR_B7 = glcm_stats(b7_u, 'SR_B7')

    # Morfología granulométrica
    open3  = swir1.focal_min(3, 'circle', 'pixels').focal_max(3, 'circle', 'pixels').rename('Open_3')
    close3 = swir1.focal_max(3, 'circle', 'pixels').focal_min(3, 'circle', 'pixels').rename('Close_3')
    gran7  = swir1.focal_max(7, 'circle', 'pixels').subtract(swir1.focal_min(7, 'circle', 'pixels')).rename('Granulo_7')

    # Textura espectral (stddev 5x5)
    kernel_5x5 = ee.Kernel.square(2)
    ndvi_std = ndvi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('NDVI_stddev_5x5')
    bsi_std  = bsi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('BSI_stddev_5x5')
    ndwi_std = ndwi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('NDWI_stddev_5x5')
    ndbi_std = ndbi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('NDBI_stddev_5x5')
    evi_std  = evi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('EVI_stddev_5x5')
    mndwi_std = mndwi.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('MNDWI_stddev_5x5')
    nbr_std  = nbr.reduceNeighborhood(ee.Reducer.stdDev(), kernel_5x5).rename('NBR_stddev_5x5')

    # Ensamblar todas las bandas
    all_bands = [
        ndvi, savi, evi, evi2, msavi, osavi, ndwi, mndwi, ndmi, ndbi, nbr, nbr2,
        psri, gci, vari, vnsir, ndsi, wdi, bsi, mbsi, gndvi, bai, emi, ri,
        fmr, clay_ratio, fe_oxide, fe_comp, clay, cri1, cri2, ari,
        nbli, tndi, tnr, bt, si_therm, shadow_index,
        tcb, tcg, tcw, tci
    ] + glcm_SR_B5 + glcm_SR_B6 + glcm_SR_B7 + [
        open3, close3, gran7,
        ndvi_std, bsi_std, ndwi_std, ndbi_std,
        evi_std, mndwi_std, nbr_std
    ]

    return img.addBands(all_bands)

def segment_snic(img: ee.Image, size=20, compact=1.5):
    seeds   = ee.Algorithms.Image.Segmentation.seedGrid(size)
    snicImg = ee.Algorithms.Image.Segmentation.SNIC(
        img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10']),
        size, compact, 8, 128, seeds)
    
    return snicImg.select('clusters').rename('label')

def segment_stats(img: ee.Image, labels: ee.Image, bands: List[str]) -> ee.Image:
    base = img.select(bands).addBands(labels)  
    return base.reduceConnectedComponents(
        reducer   = ee.Reducer.mean(),
        labelBand = 'label'                    
    )

def object_classify(img, clf, bands):
    label = segment_snic(img)
    stats = segment_stats(img, label, bands)
    return stats.classify(clf).rename('classification')

def mask_water(img: ee.Image, thr: float = 0.1) -> ee.Image:
    """Máscara simple de agua usando NDWI."""
    return img.select('NDWI').gte(thr)

def hierarchical_classify(
    img: ee.Image,
    clf: ee.Classifier,
    bands: List[str],
    water_thr: float = 0.2
) -> ee.Image:
    """
    1) Detecta agua con NDWI > thr
    2) Clasifica resto con RF
    3) Combina ambos resultados (4 = agua)
    """
    water_mask = mask_water(img, water_thr)
    water_cls  = water_mask.multiply(4).rename('classification')
    non_water  = img.updateMask(water_mask.Not())
    rf_cls     = non_water.select(bands).classify(clf).rename('classification')
    return rf_cls.unmask(water_cls)

def apply_scale_factors_l8(image: ee.Image) -> ee.Image:
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

def train_rf_objects(polygons: ee.FeatureCollection, img_ids: List[str], scale=30):
    bands = ['Clay_Index','GLCM_SR_B5_idm_5','GLCM_SR_B5_var_3','GLCM_SR_B6_var_7','NBR','TCB','TCW','GLCM_SR_B6_idm_7', 'GLCM_SR_B5_idm_7', 'GLCM_SR_B5_entropy_5','MSAVI','SAVI','GLCM_SR_B7_idm_7','GLCM_SR_B7_entropy_5', 'GLCM_SR_B6_entropy_7']



    samples = ee.FeatureCollection([])
    classes = polygons.aggregate_array('etiqueta').distinct().getInfo()
    per_class = SAMPLE_PARAMS['max_train'] // len(classes)

    for tid in img_ids:
        print("1) Iniciando get_composite para", tid)
        img   = get_composite(tid, roi=polygons.geometry())
        print("   → get_composite listo")

        print("2) Entrando a segment_snic")
        label = segment_snic(img, size=20)
        print("   → segment_snic listo")

        print("3) Calculando stats")
        stats = segment_stats(img, label, bands)
        print("   → segment_stats listo")

        print("4) Muestreo único de todos los polígonos")
        all_fc = safe_sample(
            image      = stats,
            collection = polygons,
            props      = ['etiqueta'],
            scale      = scale
        ).randomColumn('rnd', RF_PARAMS['seed'])
        print("   → safe_sample listo")

        for c in classes:
            cls_fc = all_fc.filter(ee.Filter.eq('etiqueta', c)) \
                           .limit(per_class, 'rnd')
            samples = samples.merge(cls_fc)
        print("   → Merge por clases listo")

    # 5) Barajar y dividir, limitando test a SAMPLE_PARAMS['max_test']
    print("5) Barajando y dividiendo muestras")
    samples = samples.randomColumn('rnd', RF_PARAMS['seed'])
    train = samples.filter(ee.Filter.lt('rnd', SAMPLE_PARAMS['split']))
    test  = samples.filter(ee.Filter.gte('rnd', SAMPLE_PARAMS['split'])).limit(SAMPLE_PARAMS['max_test'])
    print("   → Train/Test definidos")

    # 6) Entrenar
    print("6) Entrenando Random Forest")
    clf = ee.Classifier.smileRandomForest(**RF_PARAMS) \
            .train(train, 'etiqueta', bands)
    print("   → Modelo entrenado")

    return clf, bands

def calcular_areas_por_clase(clasificada: ee.Image, roi: ee.Geometry) -> dict:
    result = ee.Image.pixelArea().addBands(clasificada) \
        .reduceRegion(
            reducer=ee.Reducer.sum().group(1,'clase'),
            geometry=roi,
            scale=30,
            tileScale=8,
            maxPixels=1e13
        )
    groups = safe_get_info(result.get('groups'))
    return groups

def safe_sample(image, collection, props, scale, tile_scale=8, max_retries=5):
    """Intenta sampleRegions con back-off exponencial si recibe un 429."""
    wait = 10
    for attempt in range(max_retries):
        try:
            return image.sampleRegions(
                collection=collection,
                properties=props,
                scale=scale,
                tileScale=tile_scale,
                geometries=True
            )
        except EEException as e:
            if '429' in str(e):
                print(f"429 recibido, reintentando en {wait}s… (intento {attempt+1}/{max_retries})")
                time.sleep(wait)
                wait *= 2
            else:
                raise
    raise RuntimeError("Too many retries en safe_sample")
        
def safe_get_info(ee_object, retries=6, wait=5):
    """
    Intenta getInfo() con back-off exponencial si recibe un 429.
    ee_object: cualquier objeto de EE con método getInfo()
    """
    for i in range(retries):
        try:
            return ee_object.getInfo()
        except EEException as e:
            msg = str(e)
            if "429" in msg or "Too many concurrent aggregations" in msg:
                print(f" getInfo() intento {i+1}/{retries} fallido (429). Reintentando en {wait}s…")
                time.sleep(wait)
                wait *= 2
            else:
                raise
    raise RuntimeError("safe_get_info: agotados los reintentos")   

def hybrid_classify(img, clf, bands, water_thr=0.1):
    # 1) Detecta agua
    water_mask = mask_water(img, water_thr)
    water_cls  = water_mask.multiply(4).rename('classification')   # 4=agua

    # 2) Clasifica el resto por objeto
    non_water = img.updateMask(water_mask.Not())
    obj_cls   = object_classify(non_water, clf, bands).rename('classification')
    return obj_cls.unmask(water_cls)
    
    
def print_histogram(polygons):
    hist = polygons.aggregate_histogram("etiqueta")
    counts = safe_get_info(hist)
    print("Class counts:", counts)
    
def merge_polys(polys: ee.FeatureCollection,
                buf_m: float = 200) -> ee.FeatureCollection:
    # 1) buffer positivo
    buf = polys.map(lambda f: f.buffer(buf_m))

    # 2) disolver por clase
    classes = buf.aggregate_array('class').distinct()

    def dissolve_and_split(c):
        fc   = buf.filter(ee.Filter.eq('class', c))
        geom = fc.union(maxError=buf_m).geometry()          # MultiPolygon
        parts = ee.List(geom.geometries())                  # lista de polígonos
        # convierte cada parte en Feature independiente
        return ee.FeatureCollection(parts.map(
            lambda g: ee.Feature(ee.Geometry(g), {'class': c})))

    dissolved = classes.map(dissolve_and_split)             # lista de FC
    merged = ee.FeatureCollection(dissolved).flatten()      # “explota” todo
    # 3) buffer negativo para recuperar contorno aproximado
    return merged.map(lambda f: f.buffer(-buf_m))
   
def classification_to_polygons(class_img: ee.Image, roi: ee.Geometry, scale: int = 30):
    """
    Genera un FeatureCollection de polígonos contiguos.
      • geometryType='polygon'  → contornos cerrados (no líneas)
      • labelProperty='class'   → guarda el valor de la clase
    """
    polys = class_img.reduceToVectors(
        geometry      = roi,
        scale         = scale,
        eightConnected= True,
        geometryType  = 'polygon',
        labelProperty = 'class'        # nombre del atributo
    )
    return polys

    
def export_png(
    blended: ee.Image,
    img_id: str,
    roi: ee.Geometry,
    out_dir: str,
    prefix: str = 'rf',
    scale: int = 30,
    max_http_retries: int = 3,
    http_wait: int = 10
):
    coords   = roi.bounds().getInfo()['coordinates'][0]
    xs, ys   = zip(*coords)
    region   = [min(xs), min(ys), max(xs), max(ys)]
    url      = blended.getThumbURL({
        "region":     region,
        "scale":      scale,
        "format":     "png",
        "transparent": True
    })
    outfile  = os.path.join(out_dir, f"{img_id}_{prefix}.png")
    
    # 1) Intentos con requests
    for attempt in range(1, max_http_retries+1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(outfile, 'wb') as f:
                    for chunk in r.iter_content(8192):
                        if chunk:
                            f.write(chunk)
            print(f"PNG saved via HTTP: {outfile}")
            return
        except RequestException as e:
            print(f"HTTP attempt {attempt}/{max_http_retries} failed: {e}")
            if attempt < max_http_retries:
                print(f"   → Retrying in {http_wait}s…")
                time.sleep(http_wait)
                http_wait *= 2
            else:
                print(" All HTTP retries failed, falling back to Selenium")

    # 2) Fallback con Selenium
    try:
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.set_window_size(800, 800)
        driver.get(url)
        img_el = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, 'img'))
        )
        with open(outfile, 'wb') as f:
            f.write(img_el.screenshot_as_png)
        driver.quit()
        print(f"PNG saved via Selenium: {outfile}")
    except Exception as e:
        print(f"Fallback Selenium also failed: {e}")
        
        
def classify_train_ids(
    train_ids: List[str],
    clf: ee.Classifier,
    bands: List[str],
    roi: ee.Geometry,
    export_dir: str,
    prefix: str = 'L8',
    max_pixels: int = int(1e13),
    water_thr: float = 0.1
):
    """
    Clasifica únicamente los compuestos ±3 días de las escenas en train_ids.
    Usa el mismo esquema object-based (hybrid_classify) que el entrenamiento.
    """
    os.makedirs(export_dir, exist_ok=True)

    for tid in train_ids:
        print(f"\n=== Clasificando (train) {tid} ===")
        # Compuesto centrado en la fecha de la escena
        img = get_composite(tid, roi=roi)

        # Clasificación (agua por NDWI + RF por objetos)
        cls = hybrid_classify(img, clf, bands, water_thr=water_thr).rename('classification')

        # —— Export raster (Drive) ——
        task = ee.batch.Export.image.toDrive(
            image         = cls,
            description   = f'{prefix}_{tid}',
            folder        = 'GEE_exports',
            fileNamePrefix= f'{prefix}_{tid}',
            region        = roi,
            scale         = 30,
            maxPixels     = max_pixels
        )
        task.start()

        # —— Export vectores (Asset) ——
        raw_polys    = classification_to_polygons(cls, roi)
        merged_polys = merge_polys(raw_polys, buf_m=200)
        asset_path   = f'projects/ee-nataschavonsengerloeper123/assets/poligonos2/{prefix}_{tid}'
        export_asset(merged_polys, asset_id=asset_path, desc=f'{prefix}_{tid}_asset')

        # —— PNG local (control rápido) ——
        rgb = img.visualize(**_VIS_RGB).clip(roi)
        vis = cls.visualize(min=1, max=5, palette=_PALETTE, opacity=0.4).clip(roi)
        export_png(rgb.blend(vis), tid, roi, export_dir, prefix='cls')

        print(f"⮕ Tareas de export para {tid} lanzadas.")

def classify_collection(
    img_col: ee.ImageCollection,
    clf: ee.Classifier,
    bands: List[str],
    roi: ee.Geometry,
    export_dir: str,
    prefix: str = 'L8',
    composite: bool = False,
    by_year: bool = False,
    max_pixels: int = 1e13
):
    """
    Clasifica toda la colección Landsat.

    Parámetros
    ----------
    img_col      : ImageCollection ya escalada (apply_scale_factors_l8) *sin* índices.
    clf, bands   : clasificador y lista de bandas con que fue entrenado.
    roi          : Geometry para recortes y exportaciones.
    export_dir   : Carpeta local donde guardar los PNG (se crea si no existe).
    prefix       : Prefijo de los nombres de archivo.
    composite    : • False → clasifica cada escena individual
                   • True  → clasifica un median() ±3 días por órbita
    by_year      : Si True, además genera un mosaico anual median() y lo exporta.
    max_pixels   : Límite para Export.image.toDrive (sube si tu ROI es grande).
    """

    os.makedirs(export_dir, exist_ok=True)

    # 1)  Añadir índices a cada imagen de la colección
    col = img_col.map(_add_indices)


    # 3)  Clasificador → añade banda 'classification'
    def classify_img(img):
        raw   = hybrid_classify(img, clf, bands, water_thr=0.1)
        return img.addBands(raw)
        
    col_cls = col.map(classify_img)

    # 4)  Exportar raster + polígonos para cada imagen
    def export_img(img):
        img_id = img.getString('LANDSAT_ID').getInfo()
        date   = ee.Date(img.get('system:time_start')).format('YYYYMMdd').getInfo()

        # ---- Raster (GeoTIFF) ----
        task = ee.batch.Export.image.toDrive(
            image       = img.select('classification'),
            description = f'{prefix}_{img_id}_{date}',
            folder      = 'GEE_exports',
            fileNamePrefix = f'{prefix}_{img_id}_{date}',
            region      = roi,
            scale       = 30,
            maxPixels   = max_pixels
        )
        task.start()

        # ---- Contornos (SHP) ----
        raw_polys   = classification_to_polygons(img.select('classification'), roi)
        merged_polys = merge_polys(raw_polys, buf_m=200)
        
        asset_path = (f'projects/ee-nataschavonsengerloeper123/assets/poligonos2/' f'{prefix}_{img_id}_{date}')
        
        export_asset(merged_polys, asset_id = asset_path,desc = f'{prefix}_{img_id}_{date}_asset')
        

        # ---- PNG local para control rápido ----
        rgb = img.visualize(**_VIS_RGB).clip(roi)
        vis = img.select('classification').visualize(
            min=1, max=5, palette=_PALETTE, opacity=0.4).clip(roi)
        export_png(rgb.blend(vis), f'{prefix}_{img_id}_{date}',
                   roi, export_dir, prefix='cls')

        print(f'⮕  Exportando {img_id}  ({date})')

    img_list   = col_cls.toList(col_cls.size())      
    n_imgs     = col_cls.size().getInfo()           

    for i in range(n_imgs):
        img = ee.Image(img_list.get(i))             
        export_img(img)         
                    

def export_asset(fc: ee.FeatureCollection, asset_id: str, desc: str):
    """
    Exporta FeatureCollection como Asset.
    asset_id  → ruta completa: 'users/…/…'
    desc      → descripción visible en la cola de tareas
    """
    task = ee.batch.Export.table.toAsset(
        collection = fc,
        description= desc,
        assetId    = asset_id
    )
    task.start()
    print(f'➜ Export asset iniciada: {asset_id}')
 

def main():
    out_dir_png = r"C:\Users\Natascha\Desktop\Tesis\pantallazos"
    os.makedirs(out_dir_png, exist_ok=True)

    # 1) Cargar y etiquetar polígonos
    print(">>> 1) Cargando y etiquetando polígonos…")
    fc2023 = ee.FeatureCollection("projects/ee-nataschavonsengerloeper123/assets/2023").map(_add_label)
    fc2017 = ee.FeatureCollection("projects/ee-nataschavonsengerloeper123/assets/2017").map(_add_label)
    fc2014 = ee.FeatureCollection("projects/ee-nataschavonsengerloeper123/assets/20144").map(_add_label) 

    polygons = fc2014.merge(fc2023).merge(fc2017).filter(ee.Filter.gt("etiqueta", 0))
    print("    → Polígonos cargados:", safe_get_info(polygons.size()), "features")

    # 2) ROI
    print(">>> 2) Definiendo ROI (etiqueta == 5)…")
    roi_fc = polygons.filter(ee.Filter.eq("etiqueta", 5))
    roi = ee.Geometry(ee.Algorithms.If(roi_fc.size().gt(0), roi_fc.geometry(), polygons.geometry()))
    print("    → ROI OK")

    # 3) Entrenar RF (con escenas de TRAIN_IDS)
    print(">>> 3) Entrenando Random Forest…")
    unique_ids = list(dict.fromkeys(TRAIN_IDS))
    try:
        clf, bands = train_rf_objects(polygons, unique_ids)
    except Exception as e:
        print("ERROR en entrenamiento RF:", e)
        return
    print("    → RF entrenado con", len(bands), "bandas")

    # 4) Clasificar SOLO las imágenes de entrenamiento
    print(">>> 4) Clasificando SOLO TRAIN_IDS…")
    classify_train_ids(
        train_ids = unique_ids,
        clf       = clf,
        bands     = bands,
        roi       = roi,
        export_dir= out_dir_png,
        prefix    = 'L8',
        water_thr = 0.1
    )

if __name__ == '__main__':
    ee.Initialize()
    main()