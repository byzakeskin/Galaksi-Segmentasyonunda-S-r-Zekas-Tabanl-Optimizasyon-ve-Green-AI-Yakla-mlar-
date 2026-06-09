import numpy as np

class GWO:

    def __init__(self, n_wolves: int = 30, max_iter: int = 100,
                 lb: float = 0.0, ub: float = 1.0, dim: int = 2):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self.convergence_curve = []

    def optimize(self, fitness_fn):

        wolves = np.random.uniform(self.lb, self.ub,
                                   (self.n_wolves, self.dim))

        alpha_pos = np.zeros(self.dim)
        beta_pos  = np.zeros(self.dim)
        delta_pos = np.zeros(self.dim)

        alpha_score = beta_score = delta_score = float("inf")

        for iteration in range(self.max_iter):

            for i, wolf in enumerate(wolves):
                wolf = np.clip(wolf, self.lb, self.ub)
                score = fitness_fn(wolf)

                if score < alpha_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos,  beta_score  = alpha_pos.copy(), alpha_score
                    alpha_pos, alpha_score = wolf.copy(), score
                elif score < beta_score:
                    delta_pos, delta_score = beta_pos.copy(), beta_score
                    beta_pos,  beta_score  = wolf.copy(), score
                elif score < delta_score:
                    delta_pos, delta_score = wolf.copy(), score

            a = 2 - iteration * (2 / self.max_iter)

            for i in range(self.n_wolves):
                for j in range(self.dim):
                    wolves[i, j] = self._update_position(
                        wolves[i, j], a,
                        alpha_pos[j], beta_pos[j], delta_pos[j]
                    )

            self.convergence_curve.append(alpha_score)

        return alpha_pos, alpha_score

    @staticmethod
    def _update_position(x, a, alpha, beta, delta):
        def _x1(leader):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            D = abs(C * leader - x)
            return leader - A * D

        return (_x1(alpha) + _x1(beta) + _x1(delta)) / 3

class GWOThresholder:

    def __init__(self, n_thresholds: int = 2, **gwo_kwargs):
        self.n_thresholds = n_thresholds
        self.gwo = GWO(dim=n_thresholds,
                       lb=0.01, ub=0.99,
                       **gwo_kwargs)

    def _kapur_entropy(self, image_flat: np.ndarray, thresholds: np.ndarray) -> float:
        thresholds = np.sort(np.clip(thresholds, 0.01, 0.99))
        hist, _ = np.histogram(image_flat, bins=256, range=(0, 1), density=True)
        hist = hist + 1e-10  

        boundaries = [0] + [int(t * 255) for t in thresholds] + [256]
        total_entropy = 0.0

        for k in range(len(boundaries) - 1):
            segment = hist[boundaries[k]:boundaries[k + 1]]
            p_sum = segment.sum()
            if p_sum < 1e-10:
                continue
            p = segment / p_sum
            total_entropy += -np.sum(p * np.log(p + 1e-10))

        return -total_entropy 

    def threshold(self, image: np.ndarray) -> tuple:

        flat = image.flatten()
        fitness = lambda t: self._kapur_entropy(flat, t)

        best_thresholds, best_score = self.gwo.optimize(fitness)
        best_thresholds = np.sort(best_thresholds)

        mask = np.zeros_like(image, dtype=np.uint8)
        for k, t in enumerate(best_thresholds):
            mask[image >= t] = k + 1

        return mask, best_thresholds, self.gwo.convergence_curve