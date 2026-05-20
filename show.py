"""
Visualizacao do melhor tour encontrado pelo ACO sobre o dataset DataSet1.csv.

Uso:
  python show.py

Parametros configuráveis no bloco abaixo.
"""

import numpy as np
from aco import ACO

# ── Parametros configuráveis ───────────────────────────────────────────────

DATASET_PATH  = 'dataset/DataSet1.csv'
N_CITIES      = 50       # quantas cidades usar (max 300)
N_ANTS        = 20
N_ITERATIONS  = 200
ALPHA         = 1.0
BETA          = 3.0
RHO           = 0.1
Q             = 10.0
TAU0          = 0.1
TOURNAMENT    = 2
EARLY_STOP    = 30       # None para desabilitar

SAVE_TO_FILE  = True     # True: salva PNG sem abrir janela; False: exibe interativamente
OUTPUT_FILE   = 'plots/show_tour.png'

# Backend deve ser definido antes de importar pyplot
import matplotlib
matplotlib.use('Agg' if SAVE_TO_FILE else 'TkAgg')
import matplotlib.pyplot as plt

# ── Carregamento e distancias ──────────────────────────────────────────────

coords = np.loadtxt(DATASET_PATH, delimiter=',', max_rows=N_CITIES)
diff   = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
distances = np.sqrt((diff ** 2).sum(axis=2))

# ── Execucao do ACO ────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Dataset : {DATASET_PATH}  ({N_CITIES} cidades)")
    print(f"Params  : alpha={ALPHA}, beta={BETA}, rho={RHO}, Q={Q}, "
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

    tour = result['best_tour']
    dist = result['best_distance']
    print(f"Tempo        : {result['runtime']:.3f} s")
    print(f"Distancia    : {dist:.4f}")
    print(f"Iteracoes    : {result['n_iterations_run']}")

    # ── Visualizacao ──────────────────────────────────────────────────────

    # Coordenadas do tour fechado
    tour_coords = coords[tour + [tour[0]]]

    import os
    os.makedirs('plots', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- Painel esquerdo: tour sobre as cidades --
    ax = axes[0]
    ax.plot(tour_coords[:, 0], tour_coords[:, 1],
            color='steelblue', linewidth=1.0, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1],
               color='gray', s=20, zorder=2, label='Cidade')
    ax.scatter(coords[tour[0], 0], coords[tour[0], 1],
               color='green', s=80, zorder=3, label='Inicio')
    ax.scatter(coords[tour[-1], 0], coords[tour[-1], 1],
               color='red', s=80, zorder=3, label='Fim')
    ax.set_title(f'Melhor tour - {N_CITIES} cidades  (dist={dist:.4f})', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Painel direito: convergencia --
    ax2 = axes[1]
    iters = range(1, len(result['iteration_means']) + 1)
    cumulative_best = np.minimum.accumulate(result['iteration_bests'])
    ax2.plot(iters, result['iteration_means'],
             color='steelblue', alpha=0.6, linewidth=1.0,
             label='Media por iteracao')
    ax2.plot(iters, cumulative_best,
             color='crimson', linewidth=2.0,
             label='Melhor acumulado')
    ax2.set_title('Convergencia', fontsize=12)
    ax2.set_xlabel('Iteracao')
    ax2.set_ylabel('Distancia')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'ACO - TSP  |  {N_CITIES} cidades  |  {N_ANTS} formigas  |  '
                 f'alpha={ALPHA}  beta={BETA}  rho={RHO}',
                 fontsize=11)
    plt.tight_layout()

    if SAVE_TO_FILE:
        plt.savefig(OUTPUT_FILE, dpi=120)
        print(f"Imagem salva em: {OUTPUT_FILE}")
    else:
        plt.show()
