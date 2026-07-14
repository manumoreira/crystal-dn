import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_bronze_extraction(db_uri: str, anchor_date: str, periods: int = 30, freq: str = "100D", output_base_dir: str = "data/01_raw"):
    """
    Extrae snapshots históricos de la base de datos Postgres y los guarda 
    en formato Parquet, particionados por año y día en la capa Bronze (Raw).
    """
    base_path = Path(output_base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Conectando a la BD y generando {periods} snapshots hasta {anchor_date}...")
    engine = create_engine(db_uri)
    
    # Generar el rango de fechas
    snapshot_dates = pd.date_range(end=anchor_date, periods=periods, freq=freq)
    
    for current_date in snapshot_dates:
        date_str = current_date.strftime('%Y-%m-%d')
        year = current_date.year
        day_of_year = current_date.timetuple().tm_yday
        
        # Crear la partición (ej: data/01_raw/year=2026/day=126)
        partition_path = base_path / f"year={year}" / f"day={day_of_year}"
        partition_path.mkdir(parents=True, exist_ok=True)
        
        # --- 1. Extracción de Nodos ---
        node_query = f"""
            WITH target_date AS (SELECT '{date_str}'::timestamp AS t_date),
            active_releases AS (
                SELECT DISTINCT ON (shard_id) id AS release_id, shard_id
                FROM releases
                WHERE released_at <= (SELECT t_date FROM target_date)
                ORDER BY shard_id, released_at DESC
            )
            SELECT s.id AS shard_id, s.name AS node_name
            FROM active_releases ar
            JOIN shards s ON ar.shard_id = s.id;
        """
        nodes_df = pd.read_sql(node_query, engine)
        
        # --- 2. Extracción de Aristas ---
        edge_query = f"""
            WITH target_date AS (SELECT '{date_str}'::timestamp AS t_date),
            active_releases AS (
                SELECT DISTINCT ON (shard_id) id AS release_id, shard_id
                FROM releases
                WHERE released_at <= (SELECT t_date FROM target_date)
                ORDER BY shard_id, released_at DESC
            )
            SELECT s.name AS source_node, d.name AS target_node, d.spec AS raw_spec, d.scope
            FROM active_releases ar
            JOIN shards s ON ar.shard_id = s.id
            JOIN dependencies d ON ar.release_id = d.release_id
            WHERE d.name IS NOT NULL;
        """
        edges_df = pd.read_sql(edge_query, engine)

        # Filtramos las aristas para garantizar integridad referencial estricta
        #valid_nodes = set(nodes_df['node_name'])
        #edges_df = edges_df[edges_df['target_node'].isin(valid_nodes)]
        
        # --- 3. Guardado en Data Lake (Bronze) ---
        
        nodes_df['node_name'] = nodes_df['node_name'].astype(str)
        edges_df['source_node'] = edges_df['source_node'].astype(str)
        edges_df['target_node'] = edges_df['target_node'].astype(str)
        edges_df['scope'] = edges_df['scope'].astype(str)
        
        # IMPORTANTE: Usamos json.dumps en lugar de astype(str) para garantizar comillas dobles válidas
        import json
        edges_df['raw_spec'] = edges_df['raw_spec'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
        )
        
        nodes_df.to_parquet(partition_path / "nodes.parquet", index=False)
        edges_df.to_parquet(partition_path / "edges.parquet", index=False)
        
        logging.info(f"Snapshot [{date_str}] extraído: {len(nodes_df)} nodos, {len(edges_df)} aristas.")
        
        # Pausa para no saturar la base de datos local
        time.sleep(0.1)
        
    logging.info("Extracción de capa Bronze completada con éxito.")

if __name__ == "__main__":
    # Variables de entorno o configuración directa
    DB_CONNECTION_STRING = 'postgresql://postgres:postgres@localhost:5432/shardbox_development'
    ANCHOR_DATE = "2026-05-06"
    
    run_bronze_extraction(db_uri=DB_CONNECTION_STRING, anchor_date=ANCHOR_DATE)
