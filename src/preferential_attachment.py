import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter

def calculate_empirical_preferential_attachment(G_t0: nx.DiGraph, G_t1: nx.DiGraph):
    """
    Replica la metodología empírica para aislar el exponente de conexión preferencial (delta),
    midiendo la probabilidad de recibir nuevas conexiones (Pi_k) y la distribución de grados (p_k).
    
    Args:
        G_t0: Snapshot del grafo en el tiempo T0.
        G_t1: Snapshot del grafo en el tiempo T1.
        
    Returns:
        DataFrame con las métricas necesarias para ajustar log-log y encontrar tau y alpha.
    """
    # 1. Definir la población de control (AT1,T0): Paquetes que sobrevivieron
    common_nodes = set(G_t0.nodes()) & set(G_t1.nodes())
    
    # 2. Identificar aristas (dependencias) estrictamente nuevas en [T0, T1]
    edges_t0 = set(G_t0.edges())
    edges_t1 = set(G_t1.edges())
    new_edges = edges_t1 - edges_t0
    
    # 3. Filtrar conexiones nuevas cuyo destino (dependencia) esté en la población de control
    valid_new_edges = [edge for edge in new_edges if edge[1] in common_nodes]
    
    # 4. Obtener el grado (DOD - Dependency Out Degree) en T0 justo ANTES de la nueva conexión
    # Nota: En G_std (Dependiente -> Dependencia), el in_degree representa cuántos dependen de él.
    degrees_in_t0 = dict(G_t0.in_degree())
    
    # Registrar el grado 'k' que tenía cada paquete objetivo en T0
    target_k_values = [degrees_in_t0[edge[1]] for edge in valid_new_edges]
    
    # 5. Calcular Pi_k (Probabilidad empírica de recibir conexión dado un grado k)
    # Aproximado por el histograma de los grados de los paquetes a los que se unen los entrantes
    pi_k_counts = Counter(target_k_values)
    total_new_connections = len(valid_new_edges)
    
    # 6. Calcular p_k (Distribución de grados de la red en T0)
    all_k_values = [degrees_in_t0[node] for node in common_nodes]
    p_k_counts = Counter(all_k_values)
    total_common_nodes = len(common_nodes)
    
    # 7. Consolidar los datos matemáticos
    max_k = max(all_k_values) if all_k_values else 0
    results = []
    
    for k in range(max_k + 1):
        if k in p_k_counts:
            p_k = p_k_counts[k] / total_common_nodes
            pi_k = pi_k_counts.get(k, 0) / total_new_connections if total_new_connections > 0 else 0
            
            results.append({
                'k': k,
                'p_k': p_k,
                'Pi_k': pi_k
            })
            
    df_attachment = pd.DataFrame(results)
    
    # Calcular la distribución acumulada para aislar tau (mitigación de ruido)
    # Sum_Pi_k = sum(Pi_k) desde 0 hasta K
    df_attachment['Accumulated_Pi_k'] = df_attachment['Pi_k'].cumsum()
    
    # Retornar eliminando los k donde no hay nodos en la red base para evitar errores logarítmicos
    return df_attachment[df_attachment['p_k'] > 0].copy()