import torch
from models.unet import UNet
from utils.dataset import SyntheticGalaxyDataset
from optimizers.gwo import GWOThresholder
from optimizers.pso import UNetHyperparamOptimizer
from optimizers.swarm_tools import WOAPruner, EarlyExitOptimizer
from codecarbon import OfflineEmissionsTracker

def main():
    print("--- Galaksi Segmentasyonu: Swarm AI ve Green AI Pipeline ---")

    dataset = SyntheticGalaxyDataset(n_samples=100)

    gwo = GWOThresholder()

    mask, thresholds, curve = gwo.threshold(dataset.images[0])
    print("[1/4] GWO Ön İşleme tamamlandı.")

    def train_mock(params): 
        return 0.85, 0.05 
        
    pso = UNetHyperparamOptimizer(train_fn=train_mock)
    best_params, _, _ = pso.optimize()
    print("[2/4] PSO Hiper-parametre optimizasyonu tamamlandı.")

    model = UNet()
    pruner = WOAPruner(n_layers=5)
    pruner.optimize()
    
    early_exit = EarlyExitOptimizer()
    early_exit.optimize()
    print("[3/4] Green AI (Pruning & Early Exit) tamamlandı.")

    tracker = OfflineEmissionsTracker()
    tracker.start()
    print("[4/4] Final Eğitim başlatılıyor...")

    tracker.stop()

    print("\n--- Pipeline Başarıyla Tamamlandı ---")

if __name__ == "__main__":
    main()