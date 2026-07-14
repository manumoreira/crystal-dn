import time
import requests
from datetime import datetime, timezone, timedelta

# --- Configuración Constante ---
BASE_URL = "https://api.github.com"
BUS_FACTOR_THRESHOLD = 0.50  # 50% — CHAOSS default

# ── Helpers de la API ────────────────────────────────────────────────────────

def _build_headers(token: str) -> dict:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}"
    }

def _gh_get(url: str, headers: dict, params: dict = None, retries: int = 3):
    """GET con manejo de rate-limit y reintentos."""
    for attempt in range(retries):
        r = requests.get(url, headers=headers, params=params, timeout=20)
        
        if r.status_code == 202:
            # GitHub está calculando las estadísticas en background
            time.sleep(3)
            continue
            
        if r.status_code == 429 or (r.status_code == 403 and "rate limit" in r.text.lower()):
            reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time() + 2, 5)
            print(f"  [!] Rate limit alcanzado. Esperando {wait:.0f}s...")
            time.sleep(wait)
            continue
            
        r.raise_for_status()
        return r.json()
    return None

def _paginate(url: str, headers: dict, params: dict = None, max_pages: int = 20):
    """Obtiene todas las páginas de un endpoint paginado."""
    params = params or {}
    params.setdefault("per_page", 100)
    results = []
    page = 1
    
    while page <= max_pages:
        params["page"] = page
        data = _gh_get(url, headers, params)
        if not data:
            break
        results.extend(data)
        if len(data) < params["per_page"]:
            break
        page += 1
    return results

# ── Cálculos de Métricas ──────────────────────────────────────────────────────

def compute_bus_factor(contributor_stats, threshold=BUS_FACTOR_THRESHOLD):
    """Calcula el Contributor Absence Factor (Bus Factor)."""
    if not contributor_stats:
        return None, None, []

    commit_counts = []
    for c in contributor_stats:
        total = c.get("total", 0)
        if total > 0:
            login = c.get("author", {}).get("login", "unknown") if c.get("author") else "unknown"
            commit_counts.append((login, total))

    if not commit_counts:
        return None, None, []

    commit_counts.sort(key=lambda x: x[1], reverse=True)
    grand_total = sum(v for _, v in commit_counts)

    running = 0
    bus_factor = 0
    for login, count in commit_counts:
        running += count
        bus_factor += 1
        if running / grand_total >= threshold:
            break

    top_share = round(commit_counts[0][1] / grand_total * 100, 1) if commit_counts else 0

    leaderboard = [
        {"rank": i+1, "login": login, "commits": count, "share_pct": round(count / grand_total * 100, 1)}
        for i, (login, count) in enumerate(commit_counts[:10])
    ]

    return bus_factor, top_share, leaderboard

def compute_contributor_trends(owner: str, repo: str, headers: dict):
    """Obtiene el historial de commits y deriva tendencias de contribuyentes."""
    now = datetime.now(timezone.utc)
    since_1y  = (now - timedelta(days=365)).isoformat()
    since_90d = (now - timedelta(days=90)).isoformat()
    since_60d = (now - timedelta(days=60)).isoformat()
    since_30d = (now - timedelta(days=30)).isoformat()

    url = f"{BASE_URL}/repos/{owner}/{repo}/commits"
    commits_1y = _paginate(url, headers, {"since": since_1y}, max_pages=5)

    all_authors_1y = set()
    authors_90d = set()
    authors_30d = set()
    authors_prior_30d = set()
    older_authors = set()

    for c in commits_1y:
        author = None
        if c.get("author") and c["author"].get("login"):
            author = c["author"]["login"]
        elif c.get("commit", {}).get("author", {}).get("name"):
            author = c["commit"]["author"]["name"]
            
        if not author:
            continue

        date_str = c.get("commit", {}).get("author", {}).get("date", "")
        if not date_str:
            continue
            
        try:
            date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        all_authors_1y.add(author)

        if date >= datetime.fromisoformat(since_90d):
            authors_90d.add(author)
        if date >= datetime.fromisoformat(since_30d):
            authors_30d.add(author)
        if datetime.fromisoformat(since_60d) <= date < datetime.fromisoformat(since_30d):
            authors_prior_30d.add(author)
        if date < datetime.fromisoformat(since_60d):
            older_authors.add(author)

    new_contributors_30d = authors_30d - older_authors - authors_prior_30d
    contributor_growth = len(authors_30d) - len(authors_prior_30d)

    return {
        "active_contributors_90d": len(authors_90d),
        "total_contributors_1y": len(all_authors_1y),
        "new_contributors_30d": len(new_contributors_30d),
        "contributor_growth_delta": contributor_growth,
        "authors_last_30d": len(authors_30d),
        "authors_prior_30d": len(authors_prior_30d),
    }

# ── Orquestador Principal ─────────────────────────────────────────────────────

def get_chaoss_metrics(repos: list, token: str) -> list:
    """
    Punto de entrada para la notebook. 
    Itera sobre una lista de repositorios y devuelve las métricas CHAOSS.
    """
    if not token:
        raise ValueError("Se requiere un GITHUB_TOKEN válido.")
        
    headers = _build_headers(token)
    results = []
    
    for repo_full_name in repos:
        try:
            parts = repo_full_name.strip().split("/")
            if len(parts) < 2:
                continue
                
            owner, repo = parts[-2], parts[-1]
            
            # 1. Stats de Contribuyentes (Bus Factor)
            stats_url = f"{BASE_URL}/repos/{owner}/{repo}/stats/contributors"
            stats = _gh_get(stats_url, headers)
            if stats is None:
                time.sleep(4)
                stats = _gh_get(stats_url, headers)

            bus_factor, top_share, leaderboard = compute_bus_factor(stats or [])

            # 2. Tendencias de Contribuyentes (Commits)
            trends = compute_contributor_trends(owner, repo, headers)

            results.append({
                "repo": repo_full_name,
                "bus_factor": bus_factor,
                "top_contributor_share_pct": top_share,
                **trends,
                # "leaderboard_top10": leaderboard # Opcional: Descomentar si quieres ver el JSON crudo del top
            })
            
        except Exception as e:
            print(f"  [X] ERROR procesando {repo_full_name}: {e}")
            results.append({"repo": repo_full_name, "error": str(e)})

    return results