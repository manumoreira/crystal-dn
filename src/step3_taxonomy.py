import pandas as pd
import json
import re
import logging
from pathlib import Path

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classify_beta(spec):
    """
    Evalúa la especificación de versión (SemVer) de una dependencia y 
    asigna un índice Beta de vulnerabilidad a Breaking Changes.
    """
    # Lógica de clasificación extraída de tus notebooks
    if isinstance(spec, str):
        try: spec = json.loads(spec)
        except: return 0.5 
            
    if not isinstance(spec, dict): return 1.0 
    if 'branch' in spec or str(spec.get('version')).strip() == '*': return 1.0 
    if 'commit' in spec: return 0.0
    
    if 'version' in spec:
        v = str(spec['version']).strip()
        if "~>" in v or "^" in v: return 0.1  
        elif re.match(r"^[0-9]+(\.[0-9]+)*$", v): return 0.0  
        elif ">=" in v and "<" in v: return 0.1  
            
    if any(k in spec for k in ['github', 'gitlab', 'bitbucket']):
        if not any(k in spec for k in ['version', 'branch', 'commit']): return 1.0  
            
    return 0.5 

def apply_taxonomy_to_edges(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la taxonomía a un DataFrame de aristas y retorna el resultado enriquecido.
    """
    logging.info("Calculando perfiles de vulnerabilidad (Beta)...")
    
    # 1. Calcular Beta usando la lógica de clasificación
    edges_df['beta'] = edges_df['raw_spec'].apply(classify_beta)
    
    # 2. Serializar JSON a string para Parquet
    edges_df['raw_spec'] = edges_df['raw_spec'].apply(
        lambda x: json.dumps(x) if isinstance(x, dict) else str(x)
    )
    return edges_df

def process_taxonomy_layer(raw_dir: str, silver_dir: str):
    """
    Itera sobre los snapshots, usa la función apply_taxonomy_to_edges y guarda.
    """
    raw_path = Path(raw_dir)
    silver_path = Path(silver_dir)
    silver_path.mkdir(parents=True, exist_ok=True)
    
    all_edge_files = sorted(list(raw_path.rglob("edges.parquet")))
    
    for edge_file in all_edge_files:
        year_val = edge_file.parts[-3].split('=')[1]
        day_val = edge_file.parts[-2].split('=')[1]
        output_path = silver_path / f"edges_tax_{year_val}_{day_val}.parquet"
        
        if output_path.exists():
            continue
        
        # Flujo: Cargar -> Taxonomía -> Guardar
        df = pd.read_parquet(edge_file)
        df = apply_taxonomy_to_edges(df)
        df.to_parquet(output_path, index=False)
        logging.info(f"✅ Taxonomía guardada: {output_path.name}")

if __name__ == "__main__":
    process_taxonomy_layer()