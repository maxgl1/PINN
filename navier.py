import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import numpy as np

class HeatPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(3, 50)]+[nn.Linear(50, 50) for _ in range(8)]+[nn.Linear(50, 3)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
        self.apply(self._init_weights)# self.apply visite tous les modules du réseau
    
    def forward(self, x, y, t):
        u = torch.cat([x, y, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées"""
        return u # de taille (N, 1)
    
    def _init_weights(self, m):# m represente ici le module en cours d'initialisation
        if isinstance(m, nn.Linear): # pour ne pas prendre en compte les tanh qui n'ont pas de poids
            # Initialisation de Xavier pour les poids, permet de normaliser les poids initiaux
            nn.init.xavier_normal_(m.weight)
            # On initialise les biais à zéro (important pour éviter les décalages)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class Domain:
    def __init__(self,n=500, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, t_min=0, t_max=1):
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
    
    def generate_border(self):
        x_collocation = torch.linspace(self.x_min, self.x_max, self.n//2, device=device)
        y_collocation = torch.linspace(self.y_min, self.y_max, self.n//2, device=device)
        t_collocation = torch.linspace(self.t_min, self.t_max, self.n, device=device)
        self.bordure = {}

        # a gauche (inlet):
        Y, T = torch.meshgrid(y_collocation, t_collocation, indexing='ij')
        X = torch.zeros_like(Y) 
        x_inlet = X.flatten().view(-1, 1)# view casi pareil que reshape mais créé pas de copie
        y_inlet = Y.flatten().view(-1, 1)
        t_inlet = T.flatten().view(-1, 1)
        self.bordure['g'] = [x_inlet, y_inlet, t_inlet]

        # droite (pareil mais x=1)
        x_outlet = torch.ones_like(x_inlet)

        self.bordure['d'] = [x_outlet, y_inlet, t_inlet]
        
        #en haut
        X, T = torch.meshgrid(x_collocation, t_collocation, indexing='ij')
        Y = torch.ones_like(X)
        x_haut = X.flatten().view(-1, 1)# view casi pareil que reshape mais créé pas de copie
        y_haut = Y.flatten().view(-1, 1)
        t_haut = T.flatten().view(-1, 1)
        self.bordure['h'] = [x_haut, y_haut, t_haut]

        # en bas
        y_bas = torch.zeros_like(y_haut)
        self.bordure['b'] = [x_haut, y_bas, t_haut]
        return self.bordure

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
    
    def generate_no_slip(self, xc, yc, R):
        theta = torch.linspace(torch.tensor(0.0), 2*torch.pi, self.n//10)
        x_cyl = (xc + R*torch.cos(theta))
        y_cyl = (yc + R*torch.sin(theta))
        return x_cyl, y_cyl


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

    def compute_initial_condition(self, model, x, y, t):
        t_0_mask = (t<0.01).squeeze() # on créée 
        t_0 = torch.zeros_like(x[t_0_mask])
        u_0, v_0, p_0 = torch.split(model(x[t_0_mask], y[t_0_mask], t_0), 1, dim=1) # de taille (N, 1), qu'on sépare en 2 (N, 1) contient toute les position pour t=0
        
        u_target = torch.ones_like(t_0) # solution analytique
        v_target = torch.zeros_like(t_0) # solution analytique
        p_target = torch.zeros_like(t_0)
        # u_target = 10*torch.exp(-((x_0 - 0.5) ** 2 + (y_0 - 0.5) ** 2) / (2 * 0.1 ** 2)).to(device)    # Gaussienne centrée en (0.5, 0.5) avec écart-type 0.1, sur le gpu car la mse s'y fait
        mse = nn.MSELoss()
        mse_u = mse(u_0, u_target)
        mse_v = mse(v_0, v_target)
        mse_p = mse(p_0, p_target)
        return mse_u + mse_v + mse_p # on calcule la loss (valeur sous forme de tenseur direct)

    def compute_boundary_loss(self, model, bordure):# bounds contient {u:(les points, u_target)} (en float) pour chaque point de bordure ou i correspond a la dimension : ex x: sur [x, 0] ca vaut u_target
        mse = nn.MSELoss()
        s = torch.tensor(0.0, device=device)
        for k, val in bordure.items():
            if k=='g':
                u, v, p = torch.split(model(val[0], val[1], val[2]), 1, dim=1)
                u_g = torch.ones_like(u)
                v_g = torch.zeros_like(v)
                s+=mse(v, v_g)
                s+=mse(u, u_g)
            if k=='b':
                u, v, p = torch.split(model(val[0], val[1], val[2]), 1, dim=1)
                v_b = torch.zeros_like(v)
                s+=mse(v, v_b)
            if k=='h':
                u, v, p = torch.split(model(val[0], val[1], val[2]), 1, dim=1)
                v_h = torch.zeros_like(v)
                s+=mse(v_h, v)
            if k=='d':
                u, v, p = torch.split(model(val[0], val[1], val[2]), 1, dim=1)
                p_d = torch.zeros_like(u)
                s+=mse(p, p_d)
        return s 
    
    def compute_cycl_loss(self, model, x_cyl, y_cyl, t):#condition de vitesse nulle à la surface du cylindre
        t_squeezed = t.squeeze() # pour pouvoir faire le meshgrid, il faut une taille (N,) alors que t est (N, 1)

        x_t , t_t= torch.meshgrid(x_cyl, t_squeezed, indexing='ij')# on génere les x_t à tous les temps
        y_t, t_t = torch.meshgrid(y_cyl, t_squeezed, indexing='ij')
        x_t, y_t, t_t  = x_t.reshape(-1, 1), y_t.reshape(-1, 1), t_t.reshape(-1, 1)

        u, v, p = torch.split(model(x_t, y_t, t_t), 1, dim=1)

        mse = nn.MSELoss()
        mse_ucyl = mse(u, torch.zeros_like(u))
        mse_vcyl = mse(v, torch.zeros_like(v))

        return mse_ucyl + mse_vcyl

    def compute_total_loss(self, model, x, y, t, bordure, x_cyl, y_cyl, a=1, b=1, c=1):
        pde_x, pde_y, pde_cont = self.compute_pde_residual(model, x, y, t)
        ic = self.compute_initial_condition(model, x, y, t)
        bl = self.compute_boundary_loss(model, bordure)
        cyl = self.compute_cycl_loss(model, x_cyl, y_cyl, t)
        return 1*(pde_x + pde_y + pde_cont) + 10*ic + 10*bl + 10*cyl

class Solver:
    def __init__(self, model, domain, physics):
        self.losses = []
        self.model = model

        self.domain = domain
        domain.sample_2D_training()# permet de definir ce qu'il y a besoin dans domain pour la suite

        x, y, t = domain.exclude_collocation_cylinder(torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.2))
        self.x, self.y, self.t = x.to(device), y.to(device), t.to(device)# on travaille sur le gpu (mps)

        x_cyl, y_cyl = domain.generate_no_slip(torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.2))
        self.x_cyl, self.y_cyl = x_cyl.to(device), y_cyl.to(device)
        self.bordure = domain.generate_border()
        
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def train(self, num_epochs):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            #x_gpu, y_gpu, t_gpu = self.x.to(device), self.y.to(device), self.t.to(device)
            # u_pred = self.u(x_gpu, y_gpu, t_gpu)
            with torch.autocast(device_type='mps'):# permet d'optimiser les calculs (passe en float16 parfois)
                loss = self.physics.compute_total_loss(self.model, self.x, self.y, self.t, self.bordure, self.x_cyl, self.y_cyl)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
                loss.backward()

            self.optimizer.step()

            if epochs%20==0:
                print(f"epoch : {epochs}, loss = {loss.item()}")


def convert_tensor_to_array(tensor):
    array = tensor.squeeze().detach().numpy()
    return array

def plot_w_slider_2d(arr_y, domain):
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
    im = ax.imshow(evolution[:, :, 0].T, extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max]
                   , origin='lower', cmap='hot', aspect='auto', vmin=0.0, vmax=1.0)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im, ax=ax, label='Température u(x, y)')
    
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
    domain = Domain(t_max=0.5)
    domain.sample_2D_training()
    domain.exclude_collocation_cylinder(torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.1))
    model = HeatPINN().to(device) # calculs sur le gpu
    physics = Physics(0.1)
    
    solver = Solver(model, domain, physics)
    solver.train(1000)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        u, v, p = torch.split(model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device)), 1, dim=1)
    arr_u = u.squeeze().detach().cpu().numpy()
    arr_v = v.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_u, domain)

    plot_w_slider_2d(arr_v, domain)
