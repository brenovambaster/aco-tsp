"""
Experimentos ACO para o TSP

Sequencia de execucao:
  A     - Verificacao de funcionamento  (5 cidades, parametros fixos)
  B     - Influencia de Alpha e Beta    (5 cidades, 10 runs por configuracao)
  C     - Influencia da Evaporacao      (5 cidades, 10 runs por taxa)
  Final - Versao final em escala        (10, 20 e 50 cidades, early stopping)

Uso:
  python experiments.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')           # salva em arquivo sem abrir janela
import matplotlib.pyplot as plt
from collections import Counter
from aco import ACO

# ── Configuracao global ────────────────────────────────────────────────────

DATASET_PATH = 'dataset/DataSet1.csv'
PLOTS_DIR    = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════
# UTILITARIOS
# ══════════════════════════════════════════════════════════════════════════

def load_dataset(filepath, n_cities):
    """
    Carrega as primeiras n_cities linhas do CSV e retorna:
      coords    : array (n_cities, 2)  de coordenadas x,y
      distances : matriz (n_cities, n_cities) de distancias euclidianas
    """
    coords = np.loadtxt(filepath, delimiter=',', max_rows=n_cities)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return coords, distances


def compute_mode(values, decimals=3):
    """Moda por contagem de frequencias apos arredondamento."""
    rounded = [round(float(v), decimals) for v in values]
    return Counter(rounded).most_common(1)[0][0]


def print_stats(values, label):
    """Imprime media, mediana, moda e desvio padrao."""
    arr = np.array(values, dtype=float)
    print(f"  {label}:")
    print(f"    Media         : {arr.mean():.4f}")
    print(f"    Mediana       : {np.median(arr):.4f}")
    print(f"    Moda          : {compute_mode(values):.4f}")
    print(f"    Desvio padrao : {arr.std():.4f}")
    print(f"    Min / Max     : {arr.min():.4f} / {arr.max():.4f}")


def run_multiple(distances, params, n_runs=10):
    """
    Executa ACO n_runs vezes com os mesmos parametros.

    Retorna
    -------
    dist_results : list[float]  - melhor distancia de cada run
    time_results : list[float]  - tempo de execucao de cada run
    best_result  : dict         - resultado completo da melhor run
    """
    dist_results = []
    time_results = []
    best_result  = None

    for _ in range(n_runs):
        aco    = ACO(distances=distances, **params)
        result = aco.run()
        dist_results.append(result['best_distance'])
        time_results.append(result['runtime'])
        if best_result is None or result['best_distance'] < best_result['best_distance']:
            best_result = result

    return dist_results, time_results, best_result


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_convergence(iteration_means, iteration_bests, title, filename):
    """
    Grafico de convergencia com duas linhas:
      - Media por iteracao   : distancia media de todas as formigas (oscila)
      - Melhor acumulado     : running minimum do melhor por iteracao (monotonicamente decrescente)
    A linha do melhor acumulado confirma que o algoritmo esta aprendendo.
    """
    iters = range(1, len(iteration_means) + 1)
    cumulative_best = np.minimum.accumulate(iteration_bests)

    plt.figure(figsize=(9, 5))
    plt.plot(iters, iteration_means,
             linewidth=1.2, color='steelblue', alpha=0.7,
             label='Media das formigas por iteracao')
    plt.plot(iters, cumulative_best,
             linewidth=2.0, color='crimson',
             label='Melhor solucao acumulada')
    plt.title(title, fontsize=13)
    plt.xlabel('Iteracao')
    plt.ylabel('Distancia')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  -> Grafico salvo: {filename}")


def plot_boxplot(results_dict, title, filename):
    """
    Boxplots lado a lado com pontos individuais sobrepostos (jitter).
    Garante que distribuicoes degeneradas (zero variancia) sejam visiveis.
    """
    labels = list(results_dict.keys())
    data   = [results_dict[k] for k in labels]
    n      = len(labels)

    fig, ax = plt.subplots(figsize=(max(6, 2.5 * n), 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    showfliers=True)

    palette = plt.cm.Set2(np.linspace(0, 0.8, n))
    for patch, color in zip(bp['boxes'], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Pontos individuais com jitter horizontal para nao sobrepor
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        x_jitter = rng.normal(i + 1, 0.05, size=len(vals))
        ax.scatter(x_jitter, vals, color='black', alpha=0.6, s=25, zorder=3)

    # Margem extra no eixo y para que caixas degeneradas sejam visiveis
    all_vals = np.concatenate(data)
    yrange   = all_vals.max() - all_vals.min()
    margin   = max(yrange * 0.3, all_vals.mean() * 0.005)
    ax.set_ylim(all_vals.min() - margin, all_vals.max() + margin)

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Melhor distancia encontrada')
    ax.set_xlabel('Configuracao')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"  -> Grafico salvo: {filename}")


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENTO A - Verificacao de Funcionamento
# ══════════════════════════════════════════════════════════════════════════

def experiment_A():
    """
    5 cidades, 1 execucao, 30 iteracoes.
    Parametros fixos conforme especificacao.
    Verifica se o algoritmo converge (distancias diminuem ao longo das iteracoes).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO A - Verificacao de Funcionamento")
    print("5 cidades | 30 iter | alpha=1 beta=1 rho=0.03 Q=10 tau0=0.1")
    print("=" * 60)

    _, distances = load_dataset(DATASET_PATH, n_cities=5)

    # Parametros exatamente como especificado nas instrucoes
    params = dict(
        n_ants=10,
        n_iterations=30,
        alpha=1.0,
        beta=1.0,
        rho=0.03,
        Q=10.0,
        initial_pheromone=0.1,
        tournament_size=2,
        early_stopping=None,
    )

    aco    = ACO(distances=distances, **params)
    result = aco.run()

    print(f"\n  Melhor rota      : {result['best_tour']}")
    print(f"  Melhor distancia : {result['best_distance']:.4f}")
    print(f"  Iteracoes        : {result['n_iterations_run']}")
    print(f"  Tempo            : {result['runtime']:.4f} s")

    plot_convergence(
        result['iteration_means'],
        result['iteration_bests'],
        title='Experimento A - Convergencia (5 cidades, 30 iteracoes)',
        filename=os.path.join(PLOTS_DIR, 'A_convergencia.png'),
    )

    return params   # base para B e C


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENTO B - Influencia de Alpha e Beta
# ══════════════════════════════════════════════════════════════════════════

def experiment_B(base_params):
    """
    Testa (alpha=0.6, beta=0.2) versus (alpha=0.2, beta=0.6).
    10 runs por configuracao. Boxplot lado a lado.
    Retorna o melhor par (alpha, beta).
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO B - Influencia de Alpha e Beta")
    print("5 cidades | 10 runs | demais parametros = Experimento A")
    print("=" * 60)

    _, distances = load_dataset(DATASET_PATH, n_cities=5)

    configs = [
        {'alpha': 0.6, 'beta': 0.2, 'label': 'alpha=0.6, beta=0.2'},
        {'alpha': 0.2, 'beta': 0.6, 'label': 'alpha=0.2, beta=0.6'},
    ]

    results_dict  = {}
    mean_by_label = {}

    for cfg in configs:
        params    = {**base_params, 'alpha': cfg['alpha'], 'beta': cfg['beta']}
        dist_list, _, _ = run_multiple(distances, params, n_runs=10)
        label     = cfg['label']
        results_dict[label]  = dist_list
        mean_by_label[label] = float(np.mean(dist_list))
        print()
        print_stats(dist_list, label)

    plot_boxplot(
        results_dict,
        title='Experimento B - Comparacao de Alpha e Beta',
        filename=os.path.join(PLOTS_DIR, 'B_alpha_beta_boxplot.png'),
    )

    best_label = min(mean_by_label, key=mean_by_label.get)
    best_cfg   = next(c for c in configs if c['label'] == best_label)
    print(f"\n  >> Melhor configuracao selecionada: {best_label}")

    return best_cfg['alpha'], best_cfg['beta']


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENTO C - Influencia da Taxa de Evaporacao
# ══════════════════════════════════════════════════════════════════════════

def experiment_C(base_params, best_alpha, best_beta):
    """
    Testa rho in {0.01, 0.05, 0.1, 0.2} com alpha e beta do melhor resultado de B.
    10 runs por taxa. Boxplot lado a lado.
    Retorna a melhor taxa de evaporacao.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENTO C - Influencia da Taxa de Evaporacao")
    print(f"5 cidades | 10 runs | alpha={best_alpha}, beta={best_beta}")
    print("=" * 60)

    _, distances = load_dataset(DATASET_PATH, n_cities=5)

    evap_rates    = [0.01, 0.05, 0.1, 0.2]
    results_dict  = {}
    mean_by_label = {}

    for rho in evap_rates:
        params    = {**base_params, 'alpha': best_alpha, 'beta': best_beta, 'rho': rho}
        dist_list, _, _ = run_multiple(distances, params, n_runs=10)
        label     = f'rho={rho}'
        results_dict[label]  = dist_list
        mean_by_label[label] = float(np.mean(dist_list))
        print()
        print_stats(dist_list, label)

    plot_boxplot(
        results_dict,
        title='Experimento C - Taxa de Evaporacao',
        filename=os.path.join(PLOTS_DIR, 'C_evaporacao_boxplot.png'),
    )

    best_label = min(mean_by_label, key=mean_by_label.get)
    best_rho   = float(best_label.split('=')[1])
    print(f"\n  >> Melhor taxa de evaporacao selecionada: {best_rho}")

    return best_rho


# ══════════════════════════════════════════════════════════════════════════
# EXPERIMENTO FINAL - Versao Final em Escala
# ══════════════════════════════════════════════════════════════════════════

def experiment_final(best_alpha, best_beta, best_rho):
    """
    Aplica o algoritmo com os melhores parametros para 10, 20 e 50 cidades.

    Criterio de parada antecipada (early stopping): o algoritmo encerra
    quando a melhor solucao nao melhora por X=30 iteracoes consecutivas.

    10 runs por tamanho. Estatisticas e grafico de convergencia da melhor run.
    """
    EARLY_STOP_PATIENCE = 30    # x: iteracoes sem melhoria para encerrar

    print("\n" + "=" * 60)
    print("EXPERIMENTO FINAL - Versao Final em Escala")
    print(f"alpha={best_alpha} | beta={best_beta} | rho={best_rho}")
    print(f"Q=10 | tau0=0.1 | early stopping = {EARLY_STOP_PATIENCE} iteracoes")
    print("=" * 60)

    city_sizes = [10, 20, 50]

    for n in city_sizes:
        print(f"\n  {'-' * 46}")
        print(f"  {n} CIDADES - 10 runs, max 500 iteracoes")
        print(f"  {'-' * 46}")

        _, distances = load_dataset(DATASET_PATH, n_cities=n)

        params = dict(
            n_ants=20,
            n_iterations=500,
            alpha=best_alpha,
            beta=best_beta,
            rho=best_rho,
            Q=10.0,
            initial_pheromone=0.1,
            tournament_size=2,
            early_stopping=EARLY_STOP_PATIENCE,
        )

        dist_results, time_results, best_result = run_multiple(
            distances, params, n_runs=10
        )

        print()
        print_stats(dist_results, f'Distancia ({n} cidades)')
        print(f"    Tempo medio   : {np.mean(time_results):.4f} s")
        print(f"    Melhor run    : {best_result['best_distance']:.4f}")
        print(f"    Iters (melhor): {best_result['n_iterations_run']}")

        plot_convergence(
            best_result['iteration_means'],
            best_result['iteration_bests'],
            title=f'Experimento Final - {n} cidades (melhor de 10 runs)',
            filename=os.path.join(PLOTS_DIR, f'Final_{n}.png'),
        )


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  TSP com ACO - Experimentos Completos")
    print(f"  Dataset : {DATASET_PATH}")
    print("=" * 60)

    base_params           = experiment_A()
    best_alpha, best_beta = experiment_B(base_params)
    best_rho              = experiment_C(base_params, best_alpha, best_beta)
    experiment_final(best_alpha, best_beta, best_rho)

    print("\n" + "=" * 60)
    print("Todos os experimentos concluidos.")
    print(f"Graficos salvos em: {PLOTS_DIR}/")
    print("=" * 60)
