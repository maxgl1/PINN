import scipy.io 
import torch
import numpy as np

class Dataloader:
    def __init__(self, device, prop=0.1, file_name='cylinder_wake.mat'):
    # 1. Chargement (On utilise toujours scipy car PyTorch ne lit pas les .mat)
        data = scipy.io.loadmat(file_name)
        
        # 2. Conversion immédiate en Tenseurs PyTorch
        # On force souvent en float32 pour le GPU, mais on garde float64 (Double) pour la précision physique
        X_star = torch.tensor(data['X_star'], device=device, dtype=torch.float32)      # [N, 2]
        t_star = torch.tensor(data['t'], device=device, dtype=torch.float32)           # [T, 1]
        U_star = torch.tensor(data['U_star'], device=device, dtype=torch.float32)      # [N, 2, T]
        P_star = torch.tensor(data['p_star'], device=device, dtype=torch.float32)      # [N, T]
        
        # Récupération des dimensions
        N, T = X_star.shape[0], t_star.shape[0]
        
        # 3. Création de la grille (Tiling / Repeating)
        # En PyTorch, on utilise .repeat() au lieu de np.tile()
        
        # X_star[:, 0:1] garde la dimension (N, 1) au lieu de (N)
        XX = X_star[:, 0:1].repeat(1, T)           # [N, T] - Coordonnée X répétée pour chaque instant
        YY = X_star[:, 1:2].repeat(1, T)           # [N, T] - Coordonnée Y répétée pour chaque instant
        
        # Pour le temps, on répète le vecteur t (T, 1) N fois -> (T, N), puis on transpose (.t())
        TT = t_star.repeat(1, N).t()               # [N, T] - Temps répété pour chaque point
        
        # 4. Extraction des champs physiques
        UU = U_star[:, 0, :]                       # [N, T]
        VV = U_star[:, 1, :]                       # [N, T]
        PP = P_star                               # [N, T]


        # 5. Aplatissement (Flattening) pour l'entraînement
        # On utilise .reshape(-1, 1) pour obtenir une colonne de N*T lignes
        x_all = XX.reshape(-1, 1)                  # [N*T, 1]
        y_all = YY.reshape(-1, 1)                  # [N*T, 1]
        t_all = TT.reshape(-1, 1)                  # [N*T, 1]
        u_all = UU.reshape(-1, 1)                  # [N*T, 1]
        v_all = VV.reshape(-1, 1)                  # [N*T, 1]
        p_all = PP.reshape(-1, 1)                  # [N*T, 1]

        samples = int(round(N * T * prop))
        idx = torch.randperm(x_all.shape[0])[:samples]
        
        self.x_train = x_all[idx, :]
        self.y_train = y_all[idx, :]
        self.t_train = t_all[idx, :]
        self.u_train = u_all[idx, :]
        self.v_train = v_all[idx, :]
        self.p_train = p_all[idx, :]
