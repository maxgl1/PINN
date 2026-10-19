import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import scipy.io
import numpy as np
from data import Dataloader

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class NavierPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(3, 50)]+[nn.Linear(50, 50) for _ in range(8)]+[nn.Linear(50, 3)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
        self.apply(self._init_weights)# self.apply visite tous les modules du réseau
    
    def forward(self, x, y, t):
        u = torch.cat([x, y, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées
        return u # de taille (N, 1)
    
    def _init_weights(self, m):# m represente ici le module en cours d'initialisation
        if isinstance(m, nn.Linear): # pour ne pas prendre en compte les tanh qui n'ont pas de poids
            # Initialisation de Xavier pour les poids, permet de normaliser les poids initiaux
            nn.init.xavier_normal_(m.weight)
            # On initialise les biais à zéro (important pour éviter les décalages)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Domain:
    def __init__(self,n=40000, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, t_min=0, t_max=1):
        self.n = n

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.t_min = t_min
        self.t_max = t_max # nombre de points par dimensions
    
    def sample_2D_training(self):# 2D spatiale + temps
        sampler = qmc.LatinHypercube(d=3, seed=42) #2dimension (x, y) permet de mettre les points de manière uniforme (on sépare la grille en n^d blocs et on met un point dans chaque)
        sample = sampler.random(self.n).astype('float32') #on créer n points a l'interieur de [0, 1[^d, en float 32 (comme pytorch) car initialement scipy travaille en float 64, float 32 plus rapide
        l_bounds = [self.x_min, self.y_min, self.t_min]
        u_bounds = [self.x_max, self.y_max, self.t_max]
        real_points = qmc.scale(sample, l_bounds, u_bounds).astype('float32')# tableau numpy avec les bonnes bornes taille : (N, d)
        real_points_torch = torch.tensor(real_points)# ici on ne met plus le reU    
        self.x = real_points_torch[:, 0].reshape(-1, 1)# de taille (N, )
        self.y = real_points_torch[:, 1].reshape(-1, 1)
        self.t = real_points_torch[:, 2].reshape(-1, 1)# de taille (N, )
        return self.x, self.y, self.t


    def create_grid_testing(self):
        x_collocation = torch.linspace(self.x_min, self.x_max, 100)
        y_collocation = torch.linspace(self.y_min, self.y_max, 100)
        t_collocation = torch.linspace(self.t_min, self.t_max, 100)
        self.X, self.Y, self.T = torch.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij') # forme les points pour bien avoir une grille quand on plot
        self.x_grid , self.y_grid, self.t_grid = self.X.reshape(-1, 1), self.Y.reshape(-1, 1), self.T.reshape(-1, 1) # données pour notre tes(plot)
        return self.x_grid , self.y_grid, self.t_grid
    
    def exclude_collocation_cylinder(self, xc, yc, R):# bien rentrer des tenseurs

        dist = torch.sqrt((self.x-xc)**2 + (self.y-yc)**2)
        mask = dist > R #on créé le booléen pour la condition

        self.x = self.x[mask].reshape(-1, 1) #on regarde la condition pour chaque élément (mask est aussi un vecteur de booléens: on regarde si la ligne 42 de mask est vraie ou non, si vraie on gard sinon non), sans le reshape renvoie un tenseur de taille(N,)
        self.y = self.y[mask].reshape(-1, 1)
        self.t = self.t[mask].reshape(-1, 1)

        self.x.requires_grad_(True)
        self.y.requires_grad_(True)
        self.t.requires_grad_(True)
        return self.x, self.y, self.t



class Physics:
    def __init__(self, nu):
        self.nu=nu

    def compute_pde_residual(self, model, x, y, t):
        u_pred, v_pred, p_pred = torch.split(model(x, y, t), 1, dim=1) # torch.split pour séparer en 2 vecteurs (1 pour la taille des vecteurs, dim=1 pour indiquer qu'il faut couper selon les colonnes)
        
        D_u = torch.autograd.grad(u_pred, [x, y, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        # Le gradient à l'index [0] de ton résultat sera exactement la dérivée  du/dx au point(x0, t0)
        # permet de faire le bon gradient en mettant le grad_outputs (histoire de jacoienne)
        du_x, du_y , du_t = D_u[0], D_u[1], D_u[2]
        d2u_x = torch.autograd.grad(du_x, [x], grad_outputs=torch.ones_like(du_x), create_graph=True)[0] # on choppe l'element du tuple
        d2u_y = torch.autograd.grad(du_y, [y], grad_outputs=torch.ones_like(du_y), create_graph=True)[0] # on choppe l'element du tuple

        D_v = torch.autograd.grad(v_pred, [x, y, t], grad_outputs=torch.ones_like(v_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        dv_x, dv_y , dv_t = D_v[0], D_v[1], D_v[2]
        d2v_x = torch.autograd.grad(dv_x, [x], grad_outputs=torch.ones_like(dv_x), create_graph=True)[0] # on choppe l'element du tuple
        d2v_y = torch.autograd.grad(dv_y, [y], grad_outputs=torch.ones_like(dv_y), create_graph=True)[0] # on choppe l'element du tuple
        
        D_p = torch.autograd.grad(p_pred, [x, y], grad_outputs=torch.ones_like(p_pred), create_graph=True)
        dp_x , dp_y = D_p[0], D_p[1]

        residual_x = du_t + (u_pred * du_x + v_pred * du_y) + dp_x - self.nu*(d2u_x+d2u_y) # momentum x
        residual_y = dv_t + (u_pred * dv_x + v_pred*dv_y) +dp_y - self.nu*(d2v_x+d2v_y) # momentum y
        continuity = du_x + dv_y #condition incompressibilité

        
        mse=nn.MSELoss()
        mse_x = mse(residual_x, torch.zeros_like(residual_x))
        mse_y = mse(residual_y, torch.zeros_like(residual_y))
        mse_cont = mse(continuity, torch.zeros_like(continuity))

        return mse_x, mse_y, mse_cont

    def compute_data_loss(self, model, data):
        u_pred, v_pred, p_pred = torch.split(model(data.x_train, data.y_train, data.t_train), 1, dim=1)
        mse = nn.MSELoss()
        mse_data = mse(u_pred, data.u_train)+mse(v_pred, data.v_train)+mse(p_pred, data.p_train)
        return mse_data


    def compute_total_loss(self, model, x, y, t, data, a=1, b=1, c=1):
        pde_x, pde_y, pde_cont = self.compute_pde_residual(model, x, y, t)
        data = self.compute_data_loss(model, data)

        return 1*(pde_x + pde_y + pde_cont) + 1*data

class Solver:
    def __init__(self, model, domain, physics, data):
        self.losses = []
        self.model = model
        self.domain = domain
        self.data = data  

        self.x, self.y, self.t = domain.sample_2D_training()
        self.x, self.y, self.t = self.x.to(device).requires_grad_(), self.y.to(device).requires_grad_(), self.t.to(device).requires_grad_()
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)# optimiseur adam
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            'min', #on cherche à minimiser la loss
            factor=0.5, #newLr = facteur*Lr
            patience=200) # combien d'époque avant de diminuer) 
        
        ##LBFGS
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(), 
            lr=1,                       # On met souvent 1 car le line_search gère le pas
            max_iter=1000,
            tolerance_grad=1e-9,    # Nb de sous-itérations par step
            tolerance_change=1e-11,                
            history_size=50,            # Mémoire des gradients passés
            line_search_fn="strong_wolfe") # INDISPENSABLE pour la stabilité en PINNs
        
    def train(self, num_epochs):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            loss = self.physics.compute_total_loss(self.model, self.x, self.y, self.t, self.data)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step(loss.detach().item())# ici le scheduler n'a besoin que de la valeur de la loss pas du graph etc
            if epochs%10==0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"epoch : {epochs}, loss = {loss.item():3f}, LR = {current_lr:3e}")
        
        # affinage du modèle (Closure)

        def closure():
            self.lbfgs.zero_grad()
            # On recalcule la perte totale (PDE + BC + IC)
            loss = self.physics.compute_total_loss(self.model, self.x, self.y, self.t, self.data) 
            loss.backward()
            if self.iter_count % 50 == 0:
                print(f"L-BFGS Iter: {self.iter_count} | Loss: {loss.item():.6e}")
            self.iter_count += 1
            return loss
        
        self.iter_count=0
        self.lbfgs.step(closure)
        




def plot_w_slider_2d(arr_y, domain, str, normalized=True):
    # 1. Récupération des dimensions
    nx = len(np.unique(domain.x_grid.detach().numpy()))# en gros la on retrouve le nombre d'itérations qu'on a fait
    ny = len(np.unique(domain.y_grid.detach().numpy()))
    nt = len(np.unique(domain.t_grid.detach().numpy()))
    
    # 2. Reshape : Attention à l'ordre (x, y, t)
    # Si arr_y a été généré sur une grille, l'ordre doit correspondre
    evolution = arr_y.reshape(nx, ny, nt)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2)
    
    # 3. Affichage de l'état initial (t=0)
    # On affiche la tranche [:, :, 0]
    if normalized:
        im = ax.imshow(evolution[:, :, 0].T, extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max]
                   , origin='lower', cmap='hot', aspect='auto', vmin=0.0, vmax=1.0)
    if not normalized:
        im = ax.imshow(evolution[:, :, 0].T, extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max]
                   , origin='lower', cmap='hot', aspect='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, label=f'vitesse {str}(x, y)')
    
    # 4. Configuration du Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider_temps = Slider(ax_slider, 'Instant t', 0, nt - 1, valinit=0, valfmt='%d')
    
    def update(val):
        t_index = int(slider_temps.val)
        # Mise à jour des données de l'image
        im.set_data(evolution[:, :, t_index].T)
        ax.set_title(f"Équation de la navier_stockes - Pas de temps : {t_index}")
        fig.canvas.draw_idle()
    
    slider_temps.on_changed(update)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(3)
    #train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}")
    
    domain = Domain(n=40000, x_min=1, x_max=8, y_min=-2, y_max=2, t_min=0, t_max=20)
    domain.sample_2D_training()
    
    model = NavierPINN().to(device) # calculs sur le gpu
    physics = Physics(0.01)
    
    prop=0.0005
    data = Dataloader(device, prop, "cylinder_wake.mat")
    
    solver = Solver(model, domain, physics, data)
    solver.train(1000)
    
    path = f'pretrained/PINN_navier_cylinder{prop}.pth'
    torch.save(solver.model.state_dict(), path)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        u, v, p = torch.split(model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device)), 1, dim=1)
    arr_u = u.squeeze().detach().cpu().numpy()
    arr_v = v.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_u, domain, 'u')

    plot_w_slider_2d(arr_v, domain, 'v')
