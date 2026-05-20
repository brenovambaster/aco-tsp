"""
Visualizacao interativa do melhor tour ACO sobre um mapa real (OpenStreetMap).

Gera um arquivo HTML que pode ser aberto em qualquer navegador.
As coordenadas do DataSet1.csv sao abstratas (normalizadas), entao sao
mapeadas linearmente para uma regiao geografica configuravel.

Uso:
  python show_map.py
  -> abre tour_map.html no navegador automaticamente
"""

import os
import webbrowser
import numpy as np
import folium
from aco import ACO

# ── Parametros configuráveis ───────────────────────────────────────────────

DATASET_PATH = 'dataset/DataSet1.csv'
N_CITIES     = 50        # quantas cidades usar (max 300)
N_ANTS       = 20
N_ITERATIONS = 300
ALPHA        = 1.0
BETA         = 3.0
RHO          = 0.1
Q            = 10.0
TAU0         = 0.1
TOURNAMENT   = 2
EARLY_STOP   = 30

OUTPUT_HTML  = 'tour_map.html'

# Regiao geografica para mapear as coordenadas abstratas do dataset.
# Ajuste para qualquer regiao de interesse (lat/lon em graus decimais).
LAT_MIN, LAT_MAX = -23.8, -23.2   # Sul -> Norte  (ex: regiao de Sao Paulo)
LON_MIN, LON_MAX = -46.9, -46.2   # Oeste -> Leste

# ── Carregamento do dataset ────────────────────────────────────────────────

coords = np.loadtxt(DATASET_PATH, delimiter=',', max_rows=N_CITIES)

# Mapeamento linear: coordenadas abstratas [min,max] -> [LAT/LON_MIN, LAT/LON_MAX]
def scale(values, new_min, new_max):
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.full_like(values, (new_min + new_max) / 2.0)
    return new_min + (values - v_min) / (v_max - v_min) * (new_max - new_min)

lats = scale(coords[:, 1], LAT_MIN, LAT_MAX)   # y -> latitude
lons = scale(coords[:, 0], LON_MIN, LON_MAX)   # x -> longitude

# ── Execucao do ACO ────────────────────────────────────────────────────────

diff      = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=2))

print(f"Dataset  : {DATASET_PATH}  ({N_CITIES} cidades)")
print(f"Params   : alpha={ALPHA}, beta={BETA}, rho={RHO}, Q={Q}, "
      f"n_ants={N_ANTS}, n_iter={N_ITERATIONS}")
print("Rodando ACO...")

aco    = ACO(
    distances=distances,
    n_ants=N_ANTS,
    n_iterations=N_ITERATIONS,
    alpha=ALPHA,
    beta=BETA,
    rho=RHO,
    Q=Q,
    initial_pheromone=TAU0,
    tournament_size=TOURNAMENT,
    early_stopping=EARLY_STOP,
)
result = aco.run()
tour   = result['best_tour']

print(f"Tempo    : {result['runtime']:.3f} s")
print(f"Distancia: {result['best_distance']:.4f}")
print(f"Iteracoes: {result['n_iterations_run']}")

# ── Mapa folium ────────────────────────────────────────────────────────────

center_lat = (LAT_MIN + LAT_MAX) / 2
center_lon = (LON_MIN + LON_MAX) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap',
)

# Titulo flutuante no canto superior direito
title_html = f"""
<div style="position:fixed; top:10px; right:10px; z-index:1000;
            background:white; padding:10px 14px; border-radius:6px;
            box-shadow:2px 2px 6px rgba(0,0,0,0.3); font-family:Arial; font-size:13px;">
  <b>ACO - TSP</b><br>
  {N_CITIES} cidades &nbsp;|&nbsp; {N_ANTS} formigas<br>
  Distancia: <b>{result['best_distance']:.4f}</b><br>
  Iteracoes: {result['n_iterations_run']}
</div>
"""
m.get_root().html.add_child(folium.Element(title_html))

# Arestas do tour (linha azul fechada)
tour_coords = [[lats[c], lons[c]] for c in tour] + [[lats[tour[0]], lons[tour[0]]]]
folium.PolyLine(
    tour_coords,
    color='#2166ac',
    weight=2.5,
    opacity=0.8,
    tooltip='Tour otimo',
).add_to(m)

# Marcadores das cidades intermediarias
for rank, city_idx in enumerate(tour):
    lat, lon = lats[city_idx], lons[city_idx]

    if city_idx == tour[0]:
        # Cidade inicial — marcador verde maior
        folium.Marker(
            location=[lat, lon],
            tooltip=f'Inicio (cidade {city_idx})',
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
        ).add_to(m)
    elif city_idx == tour[-1]:
        # Ultima cidade antes do retorno — marcador vermelho
        folium.Marker(
            location=[lat, lon],
            tooltip=f'Ultima (cidade {city_idx})',
            icon=folium.Icon(color='red', icon='flag', prefix='fa'),
        ).add_to(m)
    else:
        # Cidades intermediarias — circulo pequeno com popup
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color='#d73027',
            fill=True,
            fill_color='#fc8d59',
            fill_opacity=0.8,
            tooltip=f'Cidade {city_idx}  (ordem: {rank + 1})',
        ).add_to(m)

# Salva e abre no navegador
m.save(OUTPUT_HTML)
print(f"\nMapa salvo em: {OUTPUT_HTML}")
print("Abrindo no navegador...")
webbrowser.open(os.path.abspath(OUTPUT_HTML))
