import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class HeatPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(3, 20)]+[nn.Linear(20, 20) for _ in range(4)]+[nn.Linear(20, 1)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
    
    def forward(self, x, y, t):
        u = torch.cat([x, y, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées
        return u # de taille (N, 1)

class Domain:
    def __init__(self,n=1000, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, t_min=0, t_max=1):
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
        real_points_torch = torch.tensor(real_points, requires_grad=True)
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

    def generate_border(self, n_bord=100):
        t_rand = torch.rand(n_bord, 1) * self.t_max# loi uniforme, on multiplie par la bonne valeur pour obtenir le bon interval
        y_rand = torch.rand(n_bord, 1) * (self.y_max - self.y_min) + self.y_min # pareil pour que ca commence a y_min
        x_rand = torch.rand(n_bord, 1) * (self.x_max - self.x_min) + self.x_min # pour que ca commence a x_min
        self.bordure = {}

        # GAUCHE (Inlet) : x=0, y=aléatoire, t=aléatoire
        self.bordure['g'] = [torch.full((n_bord, 1), self.x_min), y_rand, t_rand] #.full(size, value) : créeé n_bord point a la valeur self.x_min

        # DROITE (Outlet) : x=1, y=aléatoire, t=aléatoire
        self.bordure['d'] = [torch.full((n_bord, 1), self.x_max), y_rand, t_rand]
        
        # HAUT : x=aléatoire, y=1, t=aléatoire
        self.bordure['h'] = [x_rand, torch.full((n_bord, 1), self.y_max), t_rand]

        # BAS : x=aléatoire, y=0, t=aléatoire
        self.bordure['b'] = [x_rand, torch.full((n_bord, 1), self.y_min), t_rand]
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        for k in self.bordure:
            self.bordure[k] = [v.to(device) for v in self.bordure[k]]
        return self.bordure


class Physics:
    def __init__(self, alpha):
        self.alpha=alpha

    def compute_pde_residual(self, u, x, y, t):
        u_pred = u(x, y, t)
        D_u = torch.autograd.grad(u_pred, [x, y, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        # Le gradient à l'index [0] de ton résultat sera exactement la dérivée  du/dx au point(x0, t0)
        # permet de faire le bon gradient en mettant le grad_outputs (histoire de jacoienne)
        du_x, du_y , du_t = D_u[0], D_u[1], D_u[2]
        d2u_x = torch.autograd.grad(du_x, [x], grad_outputs=torch.ones_like(du_x), create_graph=True)[0] # on choppe l'element du tuple
        d2u_y = torch.autograd.grad(du_y, [y], grad_outputs=torch.ones_like(du_y), create_graph=True)[0] # on choppe l'element du tuple
        residual = du_t - self.alpha * (d2u_x + d2u_y)
        mse=nn.MSELoss()
        return mse(residual, torch.zeros_like(residual))

    def compute_initial_condition(self, u, x_0, y_0):
        t_0 = torch.zeros_like(x_0)
        u_0 = u(x_0, y_0, t_0) # de taille (N, 1), contient toute les position pour t=0
        u_target = torch.sin(torch.pi*x_0)*torch.sin(torch.pi*y_0).to(device) # solution analytique
        # u_target = 10*torch.exp(-((x_0 - 0.5) ** 2 + (y_0 - 0.5) ** 2) / (2 * 0.1 ** 2)).to(device)    # Gaussienne centrée en (0.5, 0.5) avec écart-type 0.1, sur le gpu car la mse s'y fait
        mse = nn.MSELoss()
        return mse(u_0, u_target) # on calcule la loss (valeur sous forme de tenseur direct)

    def compute_boundary_loss(self, u, bordure:dict, t):# bounds contient {i:(les x, u_target)} (en float) pour chaque point de bordure ou i correspond a la dimension : ex x: sur [x, 0] ca vaut u_target
        mse = nn.MSELoss()
        s=torch.tensor(0.0).to(device) # pour pouvoir sommer sur le gpu
        for k, val in bordure.items():
            if k=='g': #CL : u=0
                u_p = u(val[0], val[1], val[2])
                u_g = torch.zeros_like(u_p)
                s+=mse(u_p, u_g)
            if k=='b':# CL : u=0
                u_p = u(val[0], val[1], val[2])
                u_b = torch.zeros_like(u_p)
                s+=mse(u_p, u_b)
            if k=='h':# CL : u=0
                u_p = u(val[0], val[1], val[2])
                u_b = torch.zeros_like(u_p)
                s+=mse(u_p, u_b)
            if k=='d': #CL : u=0 
                u_p = u(val[0], val[1], val[2])
                u_b = torch.zeros_like(u_p)
                s+=mse(u_p, u_b)
        return s

    def compute_total_loss(self, u, x, y, t, bounds, a=1, b=1, c=1):
        pde = self.compute_pde_residual(u, x, y, t)
        ic = self.compute_initial_condition(u, x, y)
        bl = self.compute_boundary_loss(u, bounds, t)
        return 10*pde + 100*ic + bl

class Solver:
    def __init__(self, u, domain, physics):
        self.losses = []
        self.u = u

        self.domain = domain
        x, y, t = domain.sample_2D_training()
        self.x, self.y, self.t = x.to(device), y.to(device), t.to(device)# on travaille sur le gpu (mps)
        self.bounds = domain.generate_border()
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.u.parameters(), lr=0.01)

    def train(self, num_epochs, bounds={0.0:0.0, 1.0:0.0}):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            #x_gpu, y_gpu, t_gpu = self.x.to(device), self.y.to(device), self.t.to(device)
            # u_pred = self.u(x_gpu, y_gpu, t_gpu)
            loss = self.physics.compute_total_loss(self.u, self.x, self.y, self.t, self.bounds)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalcul plusieurs fois les memes choses)
            loss.backward()

            self.optimizer.step()
            self.losses.append(loss.item())

            if epochs%100==0:
                print(f"epoch : {epochs}, loss = {loss.item()}")




def plot_w_slider_2d(arr_y, domain):
    # 1. Récupération des dimensions
    nx = len(np.unique(domain.x_grid.detach().numpy()))# ici on retrouve le nombre d'itérations qu'on a fait
    ny = len(np.unique(domain.y_grid.detach().numpy()))
    nt = len(np.unique(domain.t_grid.detach().numpy()))
    
    # 2. Reshape : Attention à l'ordre (x, y, t)
    evolution = arr_y.reshape(nx, ny, nt)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2)
    
    # 3. Affichage de l'état initial (t=0)
    im = ax.imshow(evolution[:, :, 0].T, extent=[domain.x_min, domain.x_max, domain.y_min, domain.y_max]
                   , origin='lower', cmap='hot', aspect='auto')
    
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
        ax.set_title(f"Équation de la chaleur - Pas de temps : {t_index}")
        fig.canvas.draw_idle()
    
    slider_temps.on_changed(update)
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(3)
    #train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}")
    model = HeatPINN().to(device) # calculs sur le gpu
    physics = Physics(0.5)
    domain = Domain(t_max=0.2)
    solver = Solver(model, domain, physics)
    solver.train(2000, solver.bounds)

    path = 'pretrained/PINN_heat.pth'
    torch.save(solver.u.state_dict(), path)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        y = model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device))
    arr_y = y.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_y, domain)
