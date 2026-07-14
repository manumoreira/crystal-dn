import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_master_dataset(processed_dir: str, gold_dir: str):
    """
    Lee todas las métricas de la capa Silver, genera la fecha real 
    del snapshot y consolida un dataset maestro ordenado.
    """
    processed_path = Path(processed_dir)
    gold_path = Path(gold_dir)
    gold_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("Consolidando el Master Dataset (Capa Gold)...")
    
    all_files = list(processed_path.glob("metrics_*.parquet"))
    if not all_files:
        logging.warning("No se encontraron archivos en la capa Silver.")
        return None
        
    df_master = pd.concat((pd.read_parquet(f) for f in all_files), ignore_index=True)
    
    # Conversión de fechas (Año + Día del año -> YYYY-MM-DD)
    year_str = df_master['year'].astype(int).astype(str)
    day_str = df_master['day_of_year'].astype(int).astype(str).str.zfill(3)
    df_master['snapshot_date'] = pd.to_datetime(year_str + day_str, format='%Y%j')
    
    df_master = df_master.sort_values(by=['snapshot_date', 'pagerank'], ascending=[True, False]).reset_index(drop=True)
    
    output_path = gold_path / "ecosystem_master.parquet"
    df_master.to_parquet(output_path, index=False)
    logging.info(f"✅ Master dataset guardado. Total de registros históricos: {len(df_master)}")
    
    return df_master

def materialize_target_slices(silver_dir: str, gold_dir: str, anchor_date: str = "2026-05-06", periods: int = 30, freq: str = "100D"):
    """
    Mapea los datos enriquecidos de Silver a cortes temporales
    específicos y los materializa en la capa Gold para los modelos.
    """
    silver_path = Path(silver_dir)
    gold_path = Path(gold_dir)
    
    logging.info(f"Indexando snapshots disponibles para materializar {periods} cortes...")
    available_snapshots = {}
    
    # Buscar los archivos de aristas con la taxonomía ya aplicada en Silver
    for edge_path in silver_path.glob("edges_tax_*.parquet"):
        parts = edge_path.stem.split('_')
        year_val = parts[2]
        day_val = parts[3]
        dt = datetime.strptime(f"{year_val}-{day_val}", "%Y-%j")
        available_snapshots[dt] = edge_path

    sorted_dates = sorted(available_snapshots.keys())
    target_dates = pd.date_range(end=anchor_date, periods=periods, freq=freq)
    
    for target in target_dates:
        target_key = target.strftime('%Y-%m-%d')
        
        valid_dates = [d for d in sorted_dates if d <= target]
        
        if not valid_dates:
            logging.warning(f"⚠️ {target_key}: Omitido (No hay datos silver anteriores a esta fecha)")
            continue
            
        closest_date = valid_dates[-1]
        edge_path = available_snapshots[closest_date]
        
        parts = edge_path.stem.split('_')
        year_val = parts[2]
        day_val = parts[3]
        
        try:
            edges_df = pd.read_parquet(edge_path)
            metrics_df = pd.read_parquet(silver_path / f"metrics_{year_val}_{day_val}.parquet")
            
            # Guardado final en la capa Gold (¡ahora con el Beta incluido!)
            edges_df.to_parquet(gold_path / f"gold_edges_{target_key}.parquet", index=False)
            metrics_df.to_parquet(gold_path / f"gold_nodes_{target_key}.parquet", index=False)
            logging.info(f"✅ {target_key}: Materializado (usando datos del {closest_date.strftime('%Y-%m-%d')})")
            
        except FileNotFoundError:
            logging.error(f"❌ {target_key}: Falló (Faltan métricas Silver para {year_val}-{day_val})")

    logging.info("🏁 Capa Gold completamente poblada y lista para análisis.")

def run_gold_layer_prep():
    """Función orquestadora del Paso 3."""
    SILVER = "data/02_processed"
    GOLD = "data/03_gold"
    
    build_master_dataset(processed_dir=SILVER, gold_dir=GOLD)
    materialize_target_slices(silver_dir=SILVER, gold_dir=GOLD)

if __name__ == "__main__":
    run_gold_layer_prep()