import pandas as pd
import networkx as nx
from pathlib import Path
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_silver_processing(raw_dir: str = "data/01_raw", processed_dir: str = "data/02_processed"):
    """
    Procesa la capa Bronze (Raw), construye el grafo dirigido, 
    calcula métricas topológicas (PageRank, Centralidad, Grados) 
    y guarda los resultados en la capa Silver (Processed).
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    logging.info("⚙️ Iniciando el procesamiento por lotes de la red (Capa Silver)...")
    
    # Buscar todos los archivos de aristas (edges) extraídos en el paso 1
    all_edge_files = sorted(list(raw_path.rglob("edges.parquet")))
    
    if not all_edge_files:
        logging.warning("No se encontraron archivos en la capa Bronze. Ejecuta el paso 1 primero.")
        return

    for edge_file in all_edge_files:
        # Extraer temporalidad de la estructura de carpetas
        year_val = edge_file.parts[-3].split('=')[1]
        day_val = edge_file.parts[-2].split('=')[1]
        
        output_path = processed_path / f"metrics_{year_val}_{day_val}.parquet"
        
        # Evitar reprocesar si el snapshot ya fue calculado
        if output_path.exists():
            logging.info(f"Omitiendo Año {year_val} Día {day_val} (Ya procesado)")
            continue
            
        # --- 1. Carga de Datos ---
        edges_df = pd.read_parquet(edge_file)
        nodes_df = pd.read_parquet(edge_file.parent / "nodes.parquet")
        
        # --- 2. Construcción del Grafo e Inyección de Huérfanos ---
        G = nx.from_pandas_edgelist(
            edges_df, 
            source='source_node', 
            target='target_node', 
            create_using=nx.DiGraph()
        )
        G.add_nodes_from(nodes_df['node_name'].tolist())
        
        # --- 3. Cálculo de Métricas Matemáticas ---
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        pagerank = nx.pagerank(G)
        betweenness = nx.betweenness_centrality(G)
        
        # Cálculo del Componente Conectado Más Grande (BCC)
        if len(G) > 0:
            lcc_nodes = max(nx.weakly_connected_components(G), key=len)
        else:
            lcc_nodes = set()
            
        in_lcc = {node: (node in lcc_nodes) for node in G.nodes()}
        
        # --- 4. Consolidación y Exportación a Parquet ---
        metrics_df = pd.DataFrame({
            'node_name': list(G.nodes()),
            'year': int(year_val),
            'day_of_year': int(day_val),
            'in_degree': [in_degrees[n] for n in G.nodes()],
            'out_degree': [out_degrees[n] for n in G.nodes()],
            'pagerank': [pagerank[n] for n in G.nodes()],
            'betweenness': [betweenness[n] for n in G.nodes()],
            'is_in_lcc': [in_lcc[n] for n in G.nodes()]
        })
        
        metrics_df.to_parquet(output_path, index=False)
        logging.info(f"✅ Snapshot procesado: Año {year_val} Día {day_val} ({len(metrics_df)} nodos)")
        
    logging.info("🏁 ¡Procesamiento por lotes completado! Capa Silver completamente cargada.")

if __name__ == "__main__":
    # Si ejecutas el script directamente desde la terminal
    run_silver_processing()