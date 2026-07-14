"""
Módulo para el cálculo de métricas topológicas avanzadas del ecosistema de Crystal.
Basado en las metodologías de Decan (2018) y Zimmermann (2020).
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_reverse_dependencies(G_fail: nx.DiGraph, node: str) -> Tuple[int, int]:
    """
    Calcula las dependencias inversas directas y transitivas de un nodo.
    
    Perspectiva Decan (2018): ¿A cuántos paquetes afecta la falla de este nodo?
    Nota: Usamos G_fail donde la arista va de Dependencia -> Dependiente.
    Por lo tanto, los afectados son los sucesores y descendientes.
    
    Args:
        G_fail: Grafo direccional donde (A, B) implica que el daño viaja de A a B.
        node: Nombre del paquete a evaluar.
        
    Returns:
        (direct_reverse_deps, transitive_reverse_deps)
    """
    if node not in G_fail:
        return 0, 0
        
    direct = len(list(G_fail.successors(node)))
    # 'descendants' en G_fail nos da todos los nodos que se ven afectados transitivamente
    transitive = len(nx.descendants(G_fail, node)) 
    
    return direct, transitive

def calculate_weighted_criticality(G_fail: nx.DiGraph, pagerank_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Calcula la criticidad ponderada de cada paquete.
    
    Perspectiva Zimmermann (2020): Un paquete es crítico no solo si afecta a muchos,
    sino si afecta a los "hubs" más importantes del ecosistema.
    
    Args:
        G_fail: Grafo direccional de propagación de daño.
        pagerank_scores: Diccionario con el score de PageRank de cada nodo.
        
    Returns:
        Diccionario con la suma del PageRank de todos los paquetes afectados transitivamente.
    """
    criticality = {}
    
    for node in G_fail.nodes():
        affected_nodes = nx.descendants(G_fail, node)
        # Sumamos el PageRank de todas las "víctimas" de este nodo
        score = sum(pagerank_scores.get(affected, 0.0) for affected in affected_nodes)
        criticality[node] = score
        
    return criticality

def calculate_gini_index(array: np.ndarray) -> float:
    """
    Calcula el Índice Gini para medir la desigualdad en una distribución.
    Usado para demostrar que una minoría concentra la mayoría de dependencias inversas.
    
    Un Gini de 0 significa igualdad total. Un Gini cercano a 1 significa desigualdad extrema.
    
    Args:
        array: Array numérico (ej. cantidad de dependencias inversas por paquete).
        
    Returns:
        Valor del índice Gini (float).
    """
    array = np.array(array, dtype=np.float64)
    # Filtramos valores negativos o NaNs por seguridad
    array = array[array >= 0] 
    array += 0.0000001 # Prevenir división por cero en arrays de puros ceros
    array = np.sort(array)
    n = array.shape[0]
    if n == 0:
        return 0.0
    
    index = np.arange(1, n + 1)
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def calculate_p_impact_index(G_fail: nx.DiGraph, p_threshold_percent: float = 5.0) -> int:
    """
    Calcula el P-Impact Index (Decan, 2018).
    
    Métrica de fragilidad: El número de paquetes que, de fallar, afectarían 
    transitivamente a al menos el P% de todo el ecosistema.
    
    Args:
        G_fail: Grafo direccional de propagación de daño.
        p_threshold_percent: Porcentaje P (ej. 5.0 para el 5% del ecosistema).
        
    Returns:
        Cantidad absoluta de paquetes que superan el umbral (int).
    """
    total_nodes = G_fail.number_of_nodes()
    if total_nodes == 0:
        return 0
        
    threshold_count = (p_threshold_percent / 100.0) * total_nodes
    high_impact_packages = 0
    
    for node in G_fail.nodes():
        transitive_affected = len(nx.descendants(G_fail, node))
        if transitive_affected >= threshold_count:
            high_impact_packages += 1
            
    return high_impact_packages

def build_metrics_dataframe(G_fail: nx.DiGraph) -> pd.DataFrame:
    """
    Función de orquestación que calcula todas las métricas estáticas para todos los nodos.
    Ideal para llamar desde el Jupyter Notebook principal.
    """
    # 1. Calculamos PageRank como base para la métrica de Zimmermann
    # Usamos G_std implícito invirtiendo G_fail temporalmente para calcular autoridad
    pagerank = nx.pagerank(G_fail.reverse(), alpha=0.85)
    
    # 2. Calculamos la criticidad ponderada (Zimmermann)
    weighted_crit = calculate_weighted_criticality(G_fail, pagerank)
    
    metrics = []
    for node in G_fail.nodes():
        direct, transitive = calculate_reverse_dependencies(G_fail, node)
        metrics.append({
            'node': node,
            'direct_reverse_deps': direct,
            'transitive_reverse_deps': transitive,
            'pagerank_score': pagerank.get(node, 0),
            'weighted_criticality': weighted_crit.get(node, 0)
        })
        
    return pd.DataFrame(metrics).sort_values(by='transitive_reverse_deps', ascending=False)

def calculate_zimmermann_p_impact(G_fail: nx.DiGraph, pagerank_scores: dict, p_threshold_percent: float = 5.0) -> int:
    """
    Calcula el P-Impact Index ponderado por centralidad (Estilo Zimmermann 2020).
    
    Evalúa cuántos paquetes actúan como SPOF (Single Point of Failure) no por el 
    volumen bruto de paquetes que rompen, sino por la suma de la importancia 
    estructural (PageRank) de los paquetes afectados.
    
    Args:
        G_fail: Grafo dirigido donde las aristas van de dependencia -> dependiente.
        pagerank_scores: Diccionario precalculado con el PageRank de cada nodo.
        p_threshold_percent: Porcentaje del PageRank total de la red (ej. 5.0).
        
    Returns:
        int: Cantidad de paquetes críticos bajo este umbral ponderado.
    """
    # La suma total de PageRank suele ser 1.0, pero lo sumamos para evitar errores de redondeo
    total_network_importance = sum(pagerank_scores.values())
    critical_threshold = total_network_importance * (p_threshold_percent / 100.0)
    
    critical_packages_count = 0
    
    for node in G_fail.nodes():
        # Obtenemos todos los nodos afectados transitivamente + el nodo original
        affected_nodes = nx.descendants(G_fail, node) | {node}
        
        # Sumamos el PageRank de todos los nodos que se romperían
        impacted_importance = sum(pagerank_scores.get(n, 0) for n in affected_nodes)
        
        if impacted_importance >= critical_threshold:
            critical_packages_count += 1
            
    return critical_packages_count