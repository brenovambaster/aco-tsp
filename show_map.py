"""
Visualizacao interativa do melhor tour ACO sobre um mapa real (OpenStreetMap).

Gera um arquivo HTML que pode ser aberto em qualquer navegador.
Inclui animacao do caminho (AntPath) e lista passo a passo.

Uso:
  python show_map.py
  -> abre tour_map.html no navegador automaticamente
"""

import os
import webbrowser
import numpy as np
import folium
from folium.plugins import AntPath
from aco import ACO

# ── Parametros configuráveis ───────────────────────────────────────────────

DATASET_PATH = 'dataset/DataSet1.csv'
N_CITIES     = 20        # quantas cidades usar (max 300)
N_ANTS       = 50
N_ITERATIONS = 300
ALPHA        = 1.0
BETA         = 3.0
RHO          = 0.1
Q            = 10.0
TAU0         = 0.1
TOURNAMENT   = 3
EARLY_STOP   = 100

OUTPUT_HTML  = 'tour_map.html'

# Regiao geografica para mapear as coordenadas abstratas do dataset.
LAT_MIN, LAT_MAX = -23.8, -23.2   
LON_MIN, LON_MAX = -46.9, -46.2   

# ── Carregamento do dataset ────────────────────────────────────────────────

coords = np.loadtxt(DATASET_PATH, delimiter=',', max_rows=N_CITIES)

def scale(values, new_min, new_max):
    v_min, v_max = values.min(), values.max()
    if v_max == v_min:
        return np.full_like(values, (new_min + new_max) / 2.0)
    return new_min + (values - v_min) / (v_max - v_min) * (new_max - new_min)

lats = scale(coords[:, 1], LAT_MIN, LAT_MAX)   
lons = scale(coords[:, 0], LON_MIN, LON_MAX)   

# ── Execucao do ACO ────────────────────────────────────────────────────────

diff      = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=2))

print(f"Dataset  : {DATASET_PATH}  ({N_CITIES} cidades)")
print("Rodando ACO...")

aco = ACO(
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

# ── Mapa folium ────────────────────────────────────────────────────────────

center_lat = (LAT_MIN + LAT_MAX) / 2
center_lon = (LON_MIN + LON_MAX) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles='OpenStreetMap',
)

# Painel lateral com a lista de cidades (passo a passo)
steps_html = "<h4>Sequência de Visita</h4><ul style='list-style-type:none; padding-left:0;'>"
for i, city_idx in enumerate(tour):
    steps_html += f"<li><b>{i+1}º</b>: Cidade {city_idx}</li>"
steps_html += f"<li><b>{len(tour)+1}º</b>: Retorno à Cidade {tour[0]}</li></ul>"

info_panel_html = f"""
<div style="position:fixed; top:10px; right:10px; z-index:1000;
            background:white; padding:15px; border-radius:8px;
            box-shadow:0 0 15px rgba(0,0,0,0.2); font-family:Arial; font-size:13px;
            max-height: 80vh; overflow-y: auto; width: 200px;">
  <h3 style="margin-top:0">ACO - TSP</h3>
  <b>Distância:</b> {result['best_distance']:.4f}<br>
  <b>Cidades:</b> {N_CITIES}<br>
  <hr>
  {steps_html}
</div>
"""
m.get_root().html.add_child(folium.Element(info_panel_html))

# Arestas do tour com AntPath (animação de movimento)
tour_coords = [[lats[c], lons[c]] for c in tour] + [[lats[tour[0]], lons[tour[0]]]]
AntPath(
    locations=tour_coords,
    dash_array=[10, 20],
    delay=1000,
    color='#2166ac',
    pulse_color='#3f9',
    weight=3,
    opacity=0.8,
    tooltip='Caminho Animado'
).add_to(m)

# Marcadores das cidades
for rank, city_idx in enumerate(tour):
    lat, lon = lats[city_idx], lons[city_idx]
    
    popup_text = f"<b>Passo {rank + 1}</b><br>Cidade: {city_idx}<br>Lat: {lat:.4f}<br>Lon: {lon:.4f}"
    
    if rank == 0:
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            tooltip=f'INÍCIO (Passo 1)',
            icon=folium.Icon(color='green', icon='play', prefix='fa'),
        ).add_to(m)
    else:
        # Círculo com o número do passo
        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            color='#d73027',
            fill=True,
            fill_color='#fc8d59',
            fill_opacity=0.9,
            popup=popup_text,
            tooltip=f'Passo {rank + 1} (Cidade {city_idx})',
        ).add_to(m)

# Salva e abre no navegador
m.save(OUTPUT_HTML)
print(f"\nMapa salvo em: {OUTPUT_HTML}")
print("Abrindo no navegador...")
webbrowser.open(os.path.abspath(OUTPUT_HTML))
