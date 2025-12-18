from __future__ import annotations
import math
import os
from typing import List, Tuple
import ee
from ee import EEException
import  time, random, json
import csv
ee.Initialize()

#  PARÁMETROS

_PALETTE = [
    "#FF0000",  # Botadero (rojo)
    "#FFFF00",  # Relave   (amarillo)
    "#00FF00",  # Rajo     (verde)
    "#0000FF",  # Agua     (azul)
    "#8000FF"   # Área mina (morado)
]
_VIS_RGB = {"bands": ["SR_B4", "SR_B3", "SR_B2"], "min": 0.03, "max": 0.35, "gamma": 1.5}
RF_PARAMS = {
    "numberOfTrees": 100,
    "bagFraction": 0.8,
    "minLeafPopulation": 3,
    "maxNodes": 50,
    "seed": 42
}
SAMPLE_PARAMS = {
    "max_train": 2000,
    "max_test": 500,
    "split": 0.7
}
TRAIN_IDS = [
    "LC08_008067_20170618",
    "LC08_008067_20230705",
    "LC08_008067_20140101",
]

#  ETIQUETADO DE POLIGONOS

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

#  COMPOSITE MULTITEMPORAL 

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

#  BANDAS E INDICES

def _add_indices(img: ee.Image) -> ee.Image:
    blue, green, red = img.select('SR_B2'), img.select('SR_B3'), img.select('SR_B4')
    nir, swir1, swir2 = img.select('SR_B5'), img.select('SR_B6'), img.select('SR_B7')
    tir = img.select('ST_B10')
    eps = ee.Number(1e-6)
    
    ndvi    = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
    savi    = img.expression('((NIR-RED)*(1+L))/(NIR+RED+L)', 
               {'NIR': nir, 'RED': red, 'L': 0.5}).rename('SAVI')
    evi     = img.expression('2.5*(NIR-RED)/(NIR+6*RED-7.5*BLUE+1)', 
               {'NIR': nir, 'RED': red, 'BLUE': blue}).rename('EVI')
    ndwi    = img.normalizedDifference(['SR_B3', 'SR_B5']).rename('NDWI')
    mndwi   = img.normalizedDifference(['SR_B3', 'SR_B6']).rename('MNDWI')
    ndmi    = img.normalizedDifference(['SR_B5', 'SR_B6']).rename('NDMI')
    ndbi    = img.normalizedDifference(['SR_B6', 'SR_B5']).rename('NDBI')
    nbr     = img.normalizedDifference(['SR_B5', 'SR_B7']).rename('NBR')
    nbr2    = img.normalizedDifference(['SR_B6', 'SR_B7']).rename('NBR2')
    clay_ratio = swir1.divide(swir2.add(eps)).rename('Clay_Ratio')
    fe_oxide   = red.divide(blue.add(eps)).rename('Fe_Oxide_Index')
    fe_comp    = red.add(swir1).divide(nir.add(eps)).rename('Fe_Composite_Index')
    emi        = swir1.subtract(nir).divide(swir1.add(nir).add(eps)).rename('EMI')
    mbsi       = swir1.subtract(green).divide(swir1.add(green).add(eps)).rename('MBSI')
    bsi        = img.expression('((R+S1)-(N+B))/((R+S1)+(N+B)+eps)',
                   {'R': red, 'S1': swir1, 'N': nir, 'B': blue, 'eps': eps}).rename('BSI')
    psri       = img.expression('(RED - BLUE) / (GREEN + eps)',
                   {'RED': red, 'BLUE': blue, 'GREEN': green, 'eps': eps}).rename('PSRI')
    vari       = img.expression('(G - R)/(G + R - B)',
                   {'G': green, 'R': red, 'B': blue}).rename('VARI')
    vnsir      = red.add(green).add(blue).divide(swir1.add(swir2).add(eps)).rename('VNSIR')
    gndvi      = img.normalizedDifference(['SR_B5', 'SR_B3']).rename('GNDVI')
    bt           = tir.multiply(0.055).add(149).rename('BT')
    shadow_index = blue.add(green).divide(nir.add(eps)).rename('Shadow_Index')
    si_therm     = shadow_index.multiply(tir).rename('SI_Thermal')
    nbli         = img.normalizedDifference(['SR_B4', 'ST_B10']).rename('NBLI')
    tndi         = img.normalizedDifference(['ST_B10', 'SR_B5']).rename('TNDI')
    tnr          = tir.divide(nir.add(eps)).rename('TNR')
    ari          = nir.divide(green.add(eps)).rename('ARI')
    tcb          = img.expression('0.3029*B + 0.2786*G + 0.4733*R + 0.5599*N + 0.508*S1 + 0.1872*S2',
                   {'B': blue,'G': green,'R': red,'N': nir,'S1': swir1,'S2': swir2}).rename('TCB')
    tcg          = img.expression('-0.2941*B - 0.243*G - 0.5424*R + 0.7276*N + 0.0713*S1 - 0.1608*S2',
                   {'B': blue,'G': green,'R': red,'N': nir,'S1': swir1,'S2': swir2}).rename('TCG')
    tci          = img.expression('0.1511*B2 + 0.1973*B3 + 0.3283*B4 + 0.3407*B5 - 0.7117*B6 - 0.4559*B7',
                   {'B2': blue,'B3': green,'B4': red,'B5': nir,'B6': swir1,'B7': swir2}).rename('TCI')

    kern      = ee.Kernel.square(2)
    ndvi_std  = ndvi.reduceNeighborhood(ee.Reducer.stdDev(), kern).rename('NDVI_stddev_5x5')
    ndvi_mean = ndvi.reduceNeighborhood(ee.Reducer.mean(),   kern).rename('NDVI_mean_5x5')
    ndwi_med  = ndwi.reduceNeighborhood(ee.Reducer.median(), kern).rename('NDWI_med_5x5')
    open3  = swir1.focal_min(3, 'circle', 'pixels').focal_max(3, 'circle', 'pixels').rename('Open_3')
    close3 = swir1.focal_max(3, 'circle', 'pixels').focal_min(3, 'circle', 'pixels').rename('Close_3')
    gran7  = swir1.focal_max(7, 'circle', 'pixels') \
                .subtract(swir1.focal_min(7, 'circle', 'pixels')).rename('Granulo_7')
    def glcm_stats(band_u, name):
        stats = []
        for w in [3, 5, 7]:
            g = band_u.glcmTexture(w)
            stats += [
                g.select(f'{name}_contrast').rename(f'GLCM_{name}_contrast_{w}'),
                g.select(f'{name}_var')     .rename(f'GLCM_{name}_var_{w}'),
                g.select(f'{name}_idm')     .rename(f'GLCM_{name}_idm_{w}'),
                g.select(f'{name}_ent')     .rename(f'GLCM_{name}_ent_{w}'),
                g.select(f'{name}_diss')    .rename(f'GLCM_{name}_diss_{w}'),
                g.select(f'{name}_corr')    .rename(f'GLCM_{name}_corr_{w}'),
            ]
        return stats
    b5_u = nir.multiply(100).toUint8()
    b6_u = swir1.multiply(100).toUint8()
    b7_u = swir2.multiply(100).toUint8()
    all_bands = [
        # índices espectrales
        ndvi, savi, evi, ndwi, mndwi, ndmi, ndbi, nbr, nbr2,
        # geoquímicos / suelo fino
        clay_ratio, fe_oxide, fe_comp, emi, mbsi, bsi, psri, vari, vnsir, gndvi,
        # térmico / compuestos
        bt, shadow_index, si_therm, nbli, tndi, tnr, ari, tcb, tcg, tci,
        # texturas espectrales 5x5
        ndvi_std, ndvi_mean, ndwi_med,
        # morfología
        open3, close3, gran7
    ] + \
    glcm_stats(b5_u, 'SR_B5') + \
    glcm_stats(b6_u, 'SR_B6') + \
    glcm_stats(b7_u, 'SR_B7')
    
    return img.addBands(all_bands)

def segment_snic(img: ee.Image, size=30, compact=1):
    seeds   = ee.Algorithms.Image.Segmentation.seedGrid(size)
    snicImg = ee.Algorithms.Image.Segmentation.SNIC(
        img.select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','ST_B10']),
        size, compact, 8, 128, seeds)
    
    return snicImg.select('clusters').rename('label')

def segment_stats(img: ee.Image, labels: ee.Image, bands: List[str]) -> ee.Image:
    base = img.select(bands).addBands(labels)   # ahora contiene 'label'
    return base.reduceConnectedComponents(
        reducer   = ee.Reducer.mean(),
        labelBand = 'label'                    
    )

#  CARGA Y ESCALA
def apply_scale_factors_l8(image: ee.Image) -> ee.Image:
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    thermal = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

# ENTRENAMIENTO RF
def safe_get_info(obj, retries=1, wait=1):
    for _ in range(retries):
        try: return obj.getInfo()
        except EEException as e:
            if "429" in str(e): time.sleep(wait)
            else: raise
    raise RuntimeError("429 repetido")
    
def safe_sample(image, collection, props, scale, tile_scale=2, max_retries=5):
    wait = 2
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
            if '429' in str(e) or 'Too many concurrent aggregations' in str(e):
                time.sleep(wait)
                wait *= 2
            else:
                raise
    raise RuntimeError("safe_sample: agotados los reintentos (429)")

def train_rf_random_search(
    polygons: ee.FeatureCollection,
    img_ids: List[str],
    scale: int = 30,
    n_trials: int = 80,
    csv_path: str = "results.csv",
    min_bands: int = 10,
    max_bands: int = 25
) -> List[str]:
    # 1) Lista completa de bandas candidatas
    all_bands =rf_bands = [
    "NDVI", "SAVI", "EVI", "NDWI", "MNDWI", "NDMI", "NDBI", "NBR", "NBR2",
    "Clay_Ratio", "Fe_Oxide_Index", "Fe_Composite_Index", "EMI", "MBSI",
    "BSI", "PSRI", "VARI", "VNSIR", "GNDVI",
    "BT", "Shadow_Index", "SI_Thermal", "NBLI", "TNDI", "TNR", "ARI",
    "TCB", "TCG", "TCI",
    "NDVI_stddev_5x5", "NDVI_mean_5x5", "NDWI_med_5x5",
    "Open_3", "Close_3", "Granulo_7",  
    "GLCM_SR_B5_contrast_3", "GLCM_SR_B5_var_3", "GLCM_SR_B5_idm_3", "GLCM_SR_B5_ent_3", "GLCM_SR_B5_diss_3", "GLCM_SR_B5_corr_3",
    "GLCM_SR_B5_contrast_5", "GLCM_SR_B5_var_5", "GLCM_SR_B5_idm_5", "GLCM_SR_B5_ent_5", "GLCM_SR_B5_diss_5", "GLCM_SR_B5_corr_5",
    "GLCM_SR_B5_contrast_7", "GLCM_SR_B5_var_7", "GLCM_SR_B5_idm_7", "GLCM_SR_B5_ent_7", "GLCM_SR_B5_diss_7", "GLCM_SR_B5_corr_7",
    "GLCM_SR_B6_contrast_3", "GLCM_SR_B6_var_3", "GLCM_SR_B6_idm_3", "GLCM_SR_B6_ent_3", "GLCM_SR_B6_diss_3", "GLCM_SR_B6_corr_3",
    "GLCM_SR_B6_contrast_5", "GLCM_SR_B6_var_5", "GLCM_SR_B6_idm_5", "GLCM_SR_B6_ent_5", "GLCM_SR_B6_diss_5", "GLCM_SR_B6_corr_5",
    "GLCM_SR_B6_contrast_7", "GLCM_SR_B6_var_7", "GLCM_SR_B6_idm_7", "GLCM_SR_B6_ent_7", "GLCM_SR_B6_diss_7", "GLCM_SR_B6_corr_7",
    "GLCM_SR_B7_contrast_3", "GLCM_SR_B7_var_3", "GLCM_SR_B7_idm_3", "GLCM_SR_B7_ent_3", "GLCM_SR_B7_diss_3", "GLCM_SR_B7_corr_3",
    "GLCM_SR_B7_contrast_5", "GLCM_SR_B7_var_5", "GLCM_SR_B7_idm_5", "GLCM_SR_B7_ent_5", "GLCM_SR_B7_diss_5", "GLCM_SR_B7_corr_5",
    "GLCM_SR_B7_contrast_7", "GLCM_SR_B7_var_7", "GLCM_SR_B7_idm_7", "GLCM_SR_B7_ent_7", "GLCM_SR_B7_diss_7", "GLCM_SR_B7_corr_7",
]
    classes   = [1, 2, 3, 4, 5]
    per_class = SAMPLE_PARAMS["max_train"] // len(classes)
    samples   = ee.FeatureCollection([])

    for tid in img_ids:
        img   = get_composite(tid, roi=polygons.geometry())
        label = segment_snic(img, size=20)
        stats = segment_stats(img, label, all_bands)

        for c in classes:
            fc = safe_sample(
                stats,
                collection   = polygons.filter(ee.Filter.eq("etiqueta", c)),
                props        = ["etiqueta"],
                scale        = scale,
                tile_scale   = 16
            ).randomColumn("rnd", RF_PARAMS["seed"]) \
             .limit(per_class, "rnd")
            
            samples = samples.merge(fc)

    samples   = samples.randomColumn("rnd", RF_PARAMS["seed"])
    train     = samples.filter(ee.Filter.lt("rnd", SAMPLE_PARAMS["split"]))
    test_full = samples.filter(ee.Filter.gte("rnd", SAMPLE_PARAMS["split"]))

    # 4) Random-search y escritura de CSV
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    best_acc, best_set = -1, None

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Ahora solo un Kappa total por trial
        writer.writerow(["trial","subset_size","subset","acc_total","kappa_total"])

        best_kappa = -1
        best_set   = None

        for i in range(1, n_trials+1):
            subset_size = random.randint(min_bands, max_bands)
            subset = random.sample(all_bands, subset_size)
            print(f"\n▶️ Trial {i}/{n_trials} — size={subset_size}, bandas={subset}")
            try:
                # Entrena RF
                clf = ee.Classifier.smileRandomForest(**RF_PARAMS) \
                         .train(train, "etiqueta", subset)
                pts_all_total = ee.FeatureCollection([])
                for tid in img_ids:
                    img_full   = get_composite(tid, roi=polygons.geometry())
                    class_full = img_full.classify(clf)
                    for c in [1,2,3,4,5]:
                        pts = class_full.stratifiedSample(
                            numPoints  = SAMPLE_PARAMS["max_test"],
                            classBand  = 'classification',
                            region     = polygons.filter(ee.Filter.eq('etiqueta', c)).geometry(),
                            scale      = scale,
                            seed       = RF_PARAMS['seed'],
                            geometries = False
                        ).map(lambda f: f.set('etiqueta', c))
                        pts_all_total = pts_all_total.merge(pts)
                        
                matrix_total = pts_all_total.errorMatrix('etiqueta','classification')
                acc_total, kappa_total = compute_metrics(matrix_total)
                print(f"  • Total: Acc={acc_total:.2f}%  κ={kappa_total:.3f}")
                writer.writerow([
                    i,
                    subset_size,
                    json.dumps(subset),
                    f"{acc_total:.2f}",
                    f"{kappa_total:.3f}"
                ])
                if kappa_total > best_kappa:
                    best_kappa = kappa_total
                    best_set   = subset
            except Exception as e:
                print(f"[Trial {i}] ERROR: {e}")
            f.flush()
            time.sleep(2)

    print(f"\n Mejor trial según Kappa total: κ={best_kappa:.3f} con {len(best_set)} bandas:\n{best_set}")
    return best_set

def main():
    out_dir = r"C:\Users\dir"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Cargar y etiquetar polígonos
    fc2023 = ee.FeatureCollection(
        "projects/ee-nataschavonsengerloeper123/assets/2023"
    ).map(_add_label)
    fc2017 = ee.FeatureCollection(
        "projects/ee-nataschavonsengerloeper123/assets/2017"
    ).map(_add_label)
    fc2014 = ee.FeatureCollection(
        "projects/ee-nataschavonsengerloeper123/assets/20144"
    ).map(_add_label)

    polygons = (
        fc2014.merge(fc2023)
              .merge(fc2017)
              .filter(ee.Filter.gt("etiqueta", 0))
    )
    csv_path = os.path.join(out_dir, "random_search_results.csv")
    best_bands = train_rf_random_search(
        polygons,
        TRAIN_IDS,
        scale=60,
        n_trials=50,
        csv_path=csv_path,
        min_bands=5,
        max_bands=25
    )

    print(">>> Mejor conjunto de bandas:", best_bands)
    print(f">>> Resultados completos en: {csv_path}")

if __name__ == '__main__':
    main()
