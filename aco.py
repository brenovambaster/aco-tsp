import numpy as np
import random
import time


class ACO:
    """
    Ant Colony Optimization (AS variant) for the Traveling Salesman Problem.

    Selection method : Tournament
    Pheromone update : Classic AS  —  deposit = Q / tour_length per ant
    Evaporation      : global, applied once per iteration after all deposits

    Parameters
    ----------
    distances         : NxN numpy array of pairwise distances (pre-computed)
    n_ants            : number of ants per iteration
    n_iterations      : maximum number of iterations
    alpha             : pheromone exponent (exploitation weight)
    beta              : heuristic exponent  (greedy weight)
    rho               : evaporation rate, in (0, 1)
    Q                 : pheromone deposit coefficient  (deposit = Q / L)
    initial_pheromone : starting pheromone on every edge
    tournament_size   : number of candidates drawn in each selection step
    early_stopping    : stop if best distance unchanged for this many
                        consecutive iterations; None disables early stopping
    """

    def __init__(
        self,
        distances,
        n_ants,
        n_iterations,
        alpha,
        beta,
        rho,
        Q,
        initial_pheromone,
        tournament_size=2,
        early_stopping=None,
    ):
        self.distances = np.array(distances, dtype=float)
        self.n_cities = self.distances.shape[0]
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.tournament_size = tournament_size
        self.early_stopping = early_stopping

        # η(i,j) = 1 / d(i,j)  —  zero on the diagonal (no self-loops)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.eta = np.where(self.distances > 0, 1.0 / self.distances, 0.0)

        # τ(i,j) — pheromone matrix; diagonal is 0 (ants never revisit)
        self.tau = np.full((self.n_cities, self.n_cities), float(initial_pheromone))
        np.fill_diagonal(self.tau, 0.0)

    # ── Tour construction ──────────────────────────────────────────────────

    def _select_next(self, current, unvisited):
        """
        Tournament selection.

        Draw min(tournament_size, |unvisited|) random candidates from the
        unvisited set and return the one with the highest attractiveness
        score: τ(i,j)^alpha × η(i,j)^beta.
        """
        k = min(self.tournament_size, len(unvisited))
        candidates = random.sample(unvisited, k)

        best_city = candidates[0]
        best_score = (
            self.tau[current, candidates[0]] ** self.alpha
            * self.eta[current, candidates[0]] ** self.beta
        )
        for city in candidates[1:]:
            score = (
                self.tau[current, city] ** self.alpha
                * self.eta[current, city] ** self.beta
            )
            if score > best_score:
                best_score = score
                best_city = city

        return best_city

    def _build_tour(self):
        """Construct one complete tour for a single ant."""
        start = random.randint(0, self.n_cities - 1)
        tour = [start]
        unvisited = list(range(self.n_cities))
        unvisited.remove(start)
        while unvisited:
            nxt = self._select_next(tour[-1], unvisited)
            tour.append(nxt)
            unvisited.remove(nxt)
        return tour

    def _tour_distance(self, tour):
        n = len(tour)
        return sum(
            self.distances[tour[i], tour[(i + 1) % n]] for i in range(n)
        )

    # ── Pheromone update ───────────────────────────────────────────────────

    def _update_pheromones(self, tours, dist_list):
        """
        Classic AS pheromone update:
          1. Evaporate: τ(i,j) *= (1 - ρ)  for all edges
          2. Deposit:   τ(i,j) += Q / L_k  for each edge in ant k's tour
        """
        self.tau *= 1.0 - self.rho

        for tour, dist in zip(tours, dist_list):
            deposit = self.Q / dist
            n = len(tour)
            for i in range(n):
                a = tour[i]
                b = tour[(i + 1) % n]
                self.tau[a, b] += deposit
                self.tau[b, a] += deposit

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self):
        """
        Execute the ACO algorithm.

        Returns
        -------
        dict
            best_tour        : list[int]  — indices of the optimal tour found
            best_distance    : float
            iteration_means  : list[float] — mean ant distance per iteration
            iteration_bests  : list[float] — best ant distance per iteration
            n_iterations_run : int         — actual iterations completed
            runtime          : float       — wall-clock seconds
        """
        t0 = time.time()
        best_tour = None
        best_distance = float('inf')
        no_improve = 0
        iteration_means = []
        iteration_bests = []

        for it in range(self.n_iterations):
            tours = [self._build_tour() for _ in range(self.n_ants)]
            dist_list = [self._tour_distance(t) for t in tours]

            iter_best = min(dist_list)
            iteration_bests.append(iter_best)
            iteration_means.append(sum(dist_list) / len(dist_list))

            if iter_best < best_distance:
                best_distance = iter_best
                best_tour = tours[dist_list.index(iter_best)][:]
                no_improve = 0
            else:
                no_improve += 1

            self._update_pheromones(tours, dist_list)

            if self.early_stopping is not None and no_improve >= self.early_stopping:
                break

        return {
            'best_tour': best_tour,
            'best_distance': best_distance,
            'iteration_means': iteration_means,
            'iteration_bests': iteration_bests,
            'n_iterations_run': it + 1,
            'runtime': time.time() - t0,
        }
