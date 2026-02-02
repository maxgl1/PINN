import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import numpy as np

class HeatPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(3, 20)]+[nn.Linear(20, 20) for _ in range(4)]+[nn.Linear(20, 1)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
    
    def forward(self, x, y, t):
        u = torch.cat([x, y, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées"""
        return u # de taille (N, 1)

class Domain:
    def __init__(self,n=100, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, t_min=0, t_max=1):
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
        x_collocation = torch.linspace(self.x_min, self.x_max, self.n)
        y_collocation = torch.linspace(self.y_min, self.y_max, self.n)
        t_collocation = torch.linspace(self.t_min, self.t_max, self.n)
        self.X, self.Y, self.T = torch.meshgrid(x_collocation, y_collocation, t_collocation, indexing='ij') # forme les 2 grilles
        self.x_grid , self.y_grid, self.t_grid = self.X.reshape(-1, 1), self.Y.reshape(-1, 1), self.T.reshape(-1, 1)
        return self.x_grid , self.y_grid, self.t_grid



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
        u_target = 10*torch.exp(-((x_0 - 0.5) ** 2 + (y_0 - 0.5) ** 2) / (2 * 0.1 ** 2)).to(device)    # Gaussienne centrée en (0.5, 0.5) avec écart-type 0.1, sur le gpu car la mse s'y fait
        mse = nn.MSELoss()
        return mse(u_0, u_target) # on calcule la loss (valeur sous forme de tenseur direct)

    def compute_boundary_loss(self, u, bounds:dict, t):# bounds contient {i:(les x, u_target)} (en float) pour chaque point de bordure ou i correspond a la dimension : ex x: sur [x, 0] ca vaut u_target
        mse = nn.MSELoss()
        s=torch.tensor(0.0).to(device) # pour pouvoir sommer sur le gpu
        for k, v in bounds.items():
            if k=='x':
                x_bs = v[0]# on récupère les x
                y_bs = torch.zeros_like(x_bs)
                u_b = u(x_bs, y_bs, t)
                u_target = v[1]
                u_ts = (torch.ones_like(t)*u_target)# pareil pour avoir la forme de t sur la target
                s+=mse(u_b, u_ts)
            if k=='y':
                y_bs = v[0]# on récupère les y
                x_bs = torch.zeros_like(y_bs)
                u_b = u(x_bs, y_bs, t)
                u_target = v[1]
                u_ts = (torch.ones_like(t)*u_target)# pareil pour avoir la forme de t sur la target
                s+=mse(u_b, u_ts)
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
        self.bounds = {'x':(self.x, torch.zeros_like(self.x)), 'y':(self.y, torch.zeros_like(self.y))}
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.u.parameters(), lr=0.01)

    def train(self, num_epochs, bounds={0.0:0.0, 1.0:0.0}):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            #x_gpu, y_gpu, t_gpu = self.x.to(device), self.y.to(device), self.t.to(device)
            # u_pred = self.u(x_gpu, y_gpu, t_gpu)
            loss = self.physics.compute_total_loss(self.u, self.x, self.y, self.t, self.bounds)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
            loss.backward()

            self.optimizer.step()
            self.losses.append(loss.item())

            if epochs%20==0:
                print(f"epoch : {epochs}, loss = {loss.item()}")


def convert_tensor_to_array(tensor):
    array = tensor.squeeze().detach().numpy()
    return array

def plot_w_slider(arr_y, domain):
# Reshape en grille 2D (espace, temps)
    evolution = arr_y.reshape(domain.x_collocation.shape[0], domain.t_collocation.shape[0])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(bottom=0.25)
    
    # Plot initial
    line, = ax.plot(domain.x_collocation.detach().numpy(), evolution[:, 0])
    ax.set_xlabel('x')
    ax.set_ylabel('u(x, t)')
    ax.set_title("Équation de la chaleur")
    ax.set_ylim([evolution.min(), evolution.max()])
    
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider_temps = Slider(ax_slider, 'Instant t', 0, evolution.shape[1]-1, valinit=0, valfmt='%d')
    
    def update(val):
        t_index = int(slider_temps.val)
        line.set_ydata(evolution[:, t_index])
        fig.canvas.draw_idle()
    
    slider_temps.on_changed(update)
    plt.show()

def plot_w_slider_2d(arr_y, domain):
    # 1. Récupération des dimensions
    nx = len(np.unique(domain.X.detach().numpy()))# en gros la on retrouve le nombre d'itérations qu'on a fait
    ny = len(np.unique(domain.X.detach().numpy()))
    nt = len(np.unique(domain.X.detach().numpy()))
    
    # 2. Reshape : Attention à l'ordre (x, y, t)
    # Si arr_y a été généré sur une grille, l'ordre doit correspondre
    evolution = arr_y.reshape(nx, ny, nt)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    plt.subplots_adjust(bottom=0.2)
    
    # 3. Affichage de l'état initial (t=0)
    # On affiche la tranche [:, :, 0]
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
    torch.manual_seed(43)
    #train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}")
    model = HeatPINN().to(device) # calculs sur le gpu
    physics = Physics(0.5)
    domain = Domain()
    solver = Solver(model, domain, physics)
    solver.train(2000, solver.bounds)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        y = model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device))
    arr_y = y.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_y, domain)
    

"""
print(model.forward(torch.tensor([[3.0]]), torch.tensor([[4.0]])))
res0 = physics.compute_pde_residual(model, domain.x_grid, domain.t_grid)
print(res0)
res1 = physics.compute_initial_condition(model,torch.tensor([[1.0]]) )
print(res1)
bounds = {0.0:1.0, 1.0:5.0}
res2 = physics.compute_boundary_loss(model, bounds, torch.tensor([[0.0]]))
print(res2)
res3 = res2 = physics.compute_total_loss(model, domain.x_grid, domain.t_grid, bounds)"""
