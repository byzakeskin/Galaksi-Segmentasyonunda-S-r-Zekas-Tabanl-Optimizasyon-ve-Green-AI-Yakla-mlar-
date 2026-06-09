import numpy as np

class WOA:

    def __init__(self, n_whales: int = 20, max_iter: int = 50,
                 lb: float = 0.0, ub: float = 1.0, dim: int = 1):
        self.n_whales = n_whales
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.convergence_curve = []

    def optimize(self, fitness_fn):
        whales = np.random.uniform(self.lb, self.ub,
                                   (self.n_whales, self.dim))
        best_pos   = whales[0].copy()
        best_score = float("inf")

        for w in whales:
            s = fitness_fn(w)
            if s < best_score:
                best_score = s
                best_pos   = w.copy()

        for iteration in range(self.max_iter):
            a  = 2 - iteration * (2 / self.max_iter)   # [2, 0]
            a2 = -1 + iteration * (-1 / self.max_iter) # [-1, -2]
            b  = 1.0 

            for i in range(self.n_whales):
                p = np.random.rand()

                if p < 0.5:
                    A = 2 * a * np.random.rand(self.dim) - a
                    if np.linalg.norm(A) < 1:
                        l = (a2 - 1) * np.random.rand() + 1
                        D = np.abs(best_pos - whales[i])
                        whales[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
                    else:
                        rand_whale = whales[np.random.randint(self.n_whales)]
                        C = 2 * np.random.rand(self.dim)
                        D = np.abs(C * rand_whale - whales[i])
                        whales[i] = rand_whale - A * D
                else:
                    A = 2 * a * np.random.rand(self.dim) - a
                    C = 2 * np.random.rand(self.dim)
                    D = np.abs(C * best_pos - whales[i])
                    whales[i] = best_pos - A * D

                whales[i] = np.clip(whales[i], self.lb, self.ub)
                score = fitness_fn(whales[i])
                if score < best_score:
                    best_score = score
                    best_pos   = whales[i].copy()

            self.convergence_curve.append(best_score)

        return best_pos, best_score


class WOAPruner:

    def __init__(self, n_layers: int, energy_weight: float = 0.3, **woa_kwargs):
        self.n_layers     = n_layers
        self.energy_weight = energy_weight
        self.woa = WOA(dim=n_layers, lb=0.0, ub=0.9, **woa_kwargs)

    def _fitness(self, pruning_rates: np.ndarray,
                 base_accuracy: float = 0.90) -> float:

        mean_pr     = pruning_rates.mean()
        acc_loss    = mean_pr ** 2 * 0.15          
        energy_cost = 1.0 - mean_pr * 0.6         

        return acc_loss + self.energy_weight * energy_cost

    def optimize(self, base_accuracy: float = 0.90):
        print("\n[WOA] Model pruning optimizasyonu başlıyor...")
        fitness = lambda pr: self._fitness(pr, base_accuracy)
        best_rates, best_score = self.woa.optimize(fitness)

        print("[WOA] Optimum Budama Oranları:")
        for i, r in enumerate(best_rates):
            print(f"  Katman {i+1}: %{r*100:.1f} budama")
        print(f"[WOA] Fitness skoru: {best_score:.4f}")
        return best_rates, self.woa.convergence_curve

class FireflyAlgorithm:

    def __init__(self, n_fireflies: int = 25, max_iter: int = 50,
                 alpha: float = 0.5,   
                 beta0: float = 1.0,   
                 gamma: float = 1.0,   
                 lb: float = 0.0, ub: float = 1.0, dim: int = 1):
        self.n_fireflies = n_fireflies
        self.max_iter    = max_iter
        self.alpha       = alpha
        self.beta0       = beta0
        self.gamma       = gamma
        self.lb, self.ub = lb, ub
        self.dim         = dim
        self.convergence_curve = []

    def optimize(self, fitness_fn):
        ff = np.random.uniform(self.lb, self.ub,
                               (self.n_fireflies, self.dim))
        intensity = np.array([fitness_fn(f) for f in ff])

        best_idx   = np.argmin(intensity)
        best_pos   = ff[best_idx].copy()
        best_score = intensity[best_idx]

        for iteration in range(self.max_iter):
            alpha = self.alpha * (0.97 ** iteration)  

            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if intensity[j] < intensity[i]:
                        r2   = np.sum((ff[i] - ff[j]) ** 2)
                        beta = self.beta0 * np.exp(-self.gamma * r2)
                        noise = alpha * (np.random.rand(self.dim) - 0.5)
                        ff[i] = ff[i] + beta * (ff[j] - ff[i]) + noise
                        ff[i] = np.clip(ff[i], self.lb, self.ub)
                        intensity[i] = fitness_fn(ff[i])

            idx = np.argmin(intensity)
            if intensity[idx] < best_score:
                best_score = intensity[idx]
                best_pos   = ff[idx].copy()

            self.convergence_curve.append(best_score)

        return best_pos, best_score


class EarlyExitOptimizer:

    def __init__(self, n_exit_points: int = 3, **fa_kwargs):
        self.n_exits = n_exit_points
        self.fa = FireflyAlgorithm(dim=n_exit_points,
                                   lb=0.5, ub=0.99, **fa_kwargs)

    def _fitness(self, thresholds: np.ndarray) -> float:

        thresholds = np.sort(thresholds)
        # Enerji tasarrufu oranı (erken çıkış oranından türetilir)
        energy_saving = np.mean(1.0 - thresholds) * 0.8
        # Doğruluk kaybı (çok erken çıkış ceza)
        accuracy_loss = np.sum(np.maximum(0, 0.75 - thresholds) ** 2)
        return -energy_saving + accuracy_loss

    def optimize(self):
        print("\n[FA] Early Exit eşik optimizasyonu başlıyor...")
        best_thresh, best_score = self.fa.optimize(self._fitness)
        best_thresh = np.sort(best_thresh)

        print("[FA] Optimum Erken Çıkış Eşikleri:")
        for i, t in enumerate(best_thresh):
            print(f"  Çıkış Noktası {i+1}: güven ≥ {t:.4f}")
        print(f"[FA] Fitness skoru: {best_score:.4f}")
        return best_thresh, self.fa.convergence_curve

class ArtificialBeeColony:

    def __init__(self, n_bees: int = 20, max_iter: int = 100,
                 limit: int = 10,           # abandon eşiği
                 lb: float = 0.0, ub: float = 1.0, dim: int = 1):
        self.n_bees   = n_bees
        self.max_iter = max_iter
        self.limit    = limit
        self.lb, self.ub = lb, ub
        self.dim      = dim
        self.convergence_curve = []

    def optimize(self, fitness_fn):
        sources = np.random.uniform(self.lb, self.ub,
                                    (self.n_bees, self.dim))
        fitness  = np.array([fitness_fn(s) for s in sources])
        trials   = np.zeros(self.n_bees)

        best_idx   = np.argmin(fitness)
        best_pos   = sources[best_idx].copy()
        best_score = fitness[best_idx]

        for iteration in range(self.max_iter):
            for i in range(self.n_bees):
                k  = np.random.choice([x for x in range(self.n_bees) if x != i])
                d  = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                new_source    = sources[i].copy()
                new_source[d] = sources[i][d] + phi * (sources[i][d] - sources[k][d])
                new_source    = np.clip(new_source, self.lb, self.ub)
                new_fit       = fitness_fn(new_source)

                if new_fit < fitness[i]:
                    sources[i] = new_source
                    fitness[i] = new_fit
                    trials[i]  = 0
                else:
                    trials[i] += 1

            inv_fit = 1.0 / (1.0 + fitness - fitness.min())
            probs   = inv_fit / inv_fit.sum()

            for _ in range(self.n_bees):
                i  = np.random.choice(self.n_bees, p=probs)
                k  = np.random.choice([x for x in range(self.n_bees) if x != i])
                d  = np.random.randint(self.dim)
                phi = np.random.uniform(-1, 1)

                new_source    = sources[i].copy()
                new_source[d] = sources[i][d] + phi * (sources[i][d] - sources[k][d])
                new_source    = np.clip(new_source, self.lb, self.ub)
                new_fit       = fitness_fn(new_source)

                if new_fit < fitness[i]:
                    sources[i] = new_source
                    fitness[i] = new_fit
                    trials[i]  = 0

            for i in range(self.n_bees):
                if trials[i] > self.limit:
                    sources[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    fitness[i] = fitness_fn(sources[i])
                    trials[i]  = 0

            best_idx_now = np.argmin(fitness)
            if fitness[best_idx_now] < best_score:
                best_score = fitness[best_idx_now]
                best_pos   = sources[best_idx_now].copy()

            self.convergence_curve.append(best_score)

        return best_pos, best_score


class HardwareResourceOptimizer:

    BOUNDS = [(1, 16), (0.1, 1.0), (0.5, 4.0)]
    NAMES  = ["CPU Threads", "Memory Fraction", "GPU Batch Multiplier"]

    def __init__(self, **abc_kwargs):
        self.abc = ArtificialBeeColony(
            dim=3, lb=0, ub=1,
            **abc_kwargs
        )

    def _decode(self, pos: np.ndarray) -> dict:
        result = {}
        for i, (lb, ub) in enumerate(self.BOUNDS):
            val = lb + pos[i] * (ub - lb)
            result[self.NAMES[i]] = val
        return result

    def _fitness(self, pos: np.ndarray) -> float:

        params = self._decode(pos)
        threads  = params["CPU Threads"]
        mem_frac = params["Memory Fraction"]
        gpu_mul  = params["GPU Batch Multiplier"]

        throughput = np.log1p(threads) / np.log1p(16) * 0.5 \
                   + min(gpu_mul / 4.0, 1.0) * 0.5

        energy = threads / 16 * 0.4 + mem_frac * 0.3 + min(gpu_mul / 4, 1.0) * 0.3
        return energy - throughput

    def optimize(self):
        print("\n[ABC] Donanım kaynak optimizasyonu başlıyor...")
        best_pos, best_score = self.abc.optimize(self._fitness)
        best_params = self._decode(best_pos)

        print("[ABC] Optimum Donanım Konfigürasyonu:")
        for name, val in best_params.items():
            print(f"  {name}: {val:.2f}")
        print(f"[ABC] Fitness skoru: {best_score:.4f}")
        return best_params, self.abc.convergence_curve