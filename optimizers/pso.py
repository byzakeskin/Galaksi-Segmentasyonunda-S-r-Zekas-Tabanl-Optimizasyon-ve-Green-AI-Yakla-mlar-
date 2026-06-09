import numpy as np

class PSO:

    def __init__(self,
                 n_particles: int = 30,
                 max_iter: int = 50,
                 w: float = 0.9,        
                 w_min: float = 0.4,    
                 c1: float = 2.0,       
                 c2: float = 2.0,       
                 bounds: list = None):

        self.n_particles = n_particles
        self.max_iter = max_iter
        self.w = w
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.bounds = bounds or []
        self.dim = len(self.bounds)

        self.convergence_curve = []
        self.best_history = []

    def optimize(self, fitness_fn, verbose: bool = True):
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])

        pos = np.random.uniform(lb, ub, (self.n_particles, self.dim))
        vel = np.zeros_like(pos)

        pbest_pos   = pos.copy()
        pbest_score = np.full(self.n_particles, float("inf"))

        gbest_pos   = pos[0].copy()
        gbest_score = float("inf")

        for iteration in range(self.max_iter):
            w = self.w - (self.w - self.w_min) * iteration / self.max_iter

            for i in range(self.n_particles):
                score = fitness_fn(pos[i])

                if score < pbest_score[i]:
                    pbest_score[i] = score
                    pbest_pos[i]   = pos[i].copy()

                if score < gbest_score:
                    gbest_score = score
                    gbest_pos   = pos[i].copy()

            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)

            vel = (w * vel
                   + self.c1 * r1 * (pbest_pos - pos)
                   + self.c2 * r2 * (gbest_pos  - pos))

            pos = np.clip(pos + vel, lb, ub)

            self.convergence_curve.append(gbest_score)
            self.best_history.append(gbest_pos.copy())

            if verbose and (iteration + 1) % 10 == 0:
                print(f"  [PSO] İterasyon {iteration+1:3d}/{self.max_iter} "
                      f"| En iyi skor: {gbest_score:.6f}")

        return gbest_pos, gbest_score

class UNetHyperparamOptimizer:

    PARAM_NAMES = ["log_lr", "batch_size", "dice_weight"]

    BOUNDS = [
        (-5.0, -2.0),   
        (4.0,  32.0),   
        (0.0,   1.0),   
    ]

    def __init__(self, train_fn, **pso_kwargs):
        self.train_fn = train_fn
        self.pso = PSO(bounds=self.BOUNDS, **pso_kwargs)

    def _decode(self, pos: np.ndarray) -> dict:
        return {
            "lr":          10 ** pos[0],
            "batch_size":  max(1, int(round(pos[1]))),
            "dice_weight": float(np.clip(pos[2], 0.0, 1.0)),
            "bce_weight":  float(1.0 - np.clip(pos[2], 0.0, 1.0)),
        }

    def _fitness(self, pos: np.ndarray) -> float:
        params = self._decode(pos)
        dice_score, energy_kwh = self.train_fn(params)
        return -dice_score + 0.1 * energy_kwh

    def optimize(self, verbose: bool = True):
        print("\n[PSO] Hiper-parametre optimizasyonu başlıyor...")
        best_pos, best_score = self.pso.optimize(self._fitness, verbose=verbose)
        best_params = self._decode(best_pos)

        print("\n[PSO] Optimizasyon tamamlandı!")
        print(f"  Öğrenme Oranı  : {best_params['lr']:.6f}")
        print(f"  Batch Size     : {best_params['batch_size']}")
        print(f"  Dice Ağırlığı  : {best_params['dice_weight']:.4f}")
        print(f"  BCE  Ağırlığı  : {best_params['bce_weight']:.4f}")
        print(f"  En İyi Fitness : {best_score:.6f}")

        return best_params, best_score, self.pso.convergence_curve