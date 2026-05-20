"""
Gera um GIF animado mostrando o passo a passo da melhor rota encontrada pelo ACO.
Utiliza Matplotlib Animation.

Uso:
  python show_gif.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
from aco import ACO

# ── Parametros configuráveis ───────────────────────────────────────────────

DATASET_PATH  = 'dataset/DataSet1.csv'
N_CITIES      = 20       # Reduzido para o GIF ficar leve e claro
N_ANTS        = 20
N_ITERATIONS  = 200
ALPHA         = 1.0
BETA          = 3.0
RHO           = 0.1
Q             = 10.0
TAU0          = 0.1
TOURNAMENT    = 2
EARLY_STOP    = 30

OUTPUT_GIF    = 'assets/path_trace_new.gif'

# ── Execucao do ACO ────────────────────────────────────────────────────────

print(f"Dataset : {DATASET_PATH}  ({N_CITIES} cidades)")
coords = np.loadtxt(DATASET_PATH, delimiter=',', max_rows=N_CITIES)
diff   = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=2))

print("Rodando ACO para encontrar a melhor rota...")
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
tour = result['best_tour']
dist = result['best_distance']

# Adiciona o retorno à cidade inicial para fechar o ciclo
tour_closed = tour + [tour[0]]
path_coords = coords[tour_closed]

# ── Animacao ───────────────────────────────────────────────────────────────

print("Gerando animação...")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(coords[:, 0].min() - 0.05, coords[:, 0].max() + 0.05)
ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
ax.set_title(f"ACO TSP - Passo a Passo (Dist: {dist:.4f})")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.grid(True, linestyle='--', alpha=0.6)

# Desenha todas as cidades em cinza ao fundo
ax.scatter(coords[:, 0], coords[:, 1], c='gray', s=30, zorder=1, alpha=0.5)

# Elementos que serão atualizados
line, = ax.plot([], [], 'steelblue', lw=2, zorder=2)
current_point = ax.scatter([], [], c='red', s=100, zorder=4)
visited_points = ax.scatter([], [], c='green', s=50, zorder=3)

# Texto de status
status_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.7))

def init():
    line.set_data([], [])
    current_point.set_offsets(np.empty((0, 2)))
    visited_points.set_offsets(np.empty((0, 2)))
    status_text.set_text('')
    return line, current_point, visited_points, status_text

def update(frame):
    # frame vai de 0 até len(tour_closed) - 1
    current_path = path_coords[:frame+1]
    
    # Desenha a linha até o momento
    line.set_data(current_path[:, 0], current_path[:, 1])
    
    # Marcador da cidade atual
    if frame < len(tour_closed):
        curr_idx = tour_closed[frame]
        current_point.set_offsets([coords[curr_idx]])
        
        # Marcadores das cidades já visitadas
        visited_indices = tour[:frame]
        if visited_indices:
            visited_points.set_offsets(coords[visited_indices])
        
        status_text.set_text(f"Passo: {frame}/{len(tour)}\nCidade Atual: {curr_idx}")
    
    return line, current_point, visited_points, status_text

# Cria a animação
# frames: um frame para cada cidade + o fechamento do tour
ani = FuncAnimation(fig, update, frames=len(tour_closed), init_func=init, 
                    blit=True, interval=300, repeat=True)

# Salva o GIF
os.makedirs('assets', exist_ok=True)
try:
    writer = PillowWriter(fps=5)
    ani.save(OUTPUT_GIF, writer=writer)
    print(f"Sucesso! GIF salvo em: {OUTPUT_GIF}")
except Exception as e:
    print(f"Erro ao salvar GIF: {e}")
    print("Certifique-se de que a biblioteca 'Pillow' está instalada: pip install Pillow")

plt.close()
