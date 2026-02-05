import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import numpy as np
from main import Domain

class HeatPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(3, 20)]+[nn.Linear(20, 20) for _ in range(4)]+[nn.Linear(20, 2)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
    
    def forward(self, x, y, t):
        u = torch.cat([x, y, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées"""
        return u # de taille (N, 1)



class Physics:
    def __init__(self, nu):
        self.nu=nu

    def compute_pde_residual(self, model, x, y, t):
        u_pred, v_pred = torch.split(model(x, y, t), 1, dim=1) # torch.split pour séparer en 2 vecteurs (1 pour la taille des vecteurs, dim=1 pour indiquer qu'il faut couper selon les colonnes)
        
        D_u = torch.autograd.grad(u_pred, [x, y, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        # Le gradient à l'index [0] de ton résultat sera exactement la dérivée  du/dx au point(x0, t0)
        # permet de faire le bon gradient en mettant le grad_outputs (histoire de jacoienne)
        du_x, du_y , du_t = D_u[0], D_u[1], D_u[2]
        d2u_x = torch.autograd.grad(du_x, [x], grad_outputs=torch.ones_like(du_x), create_graph=True)[0] # on choppe l'element du tuple
        d2u_y = torch.autograd.grad(du_y, [y], grad_outputs=torch.ones_like(du_y), create_graph=True)[0] # on choppe l'element du tuple

        D_v = torch.autograd.grad(u_pred, [x, y, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        dv_x, dv_y , dv_t = D_v[0], D_v[1], D_v[2]
        d2v_x = torch.autograd.grad(dv_x, [x], grad_outputs=torch.ones_like(dv_x), create_graph=True)[0] # on choppe l'element du tuple
        d2v_y = torch.autograd.grad(dv_y, [y], grad_outputs=torch.ones_like(dv_y), create_graph=True)[0] # on choppe l'element du tuple
        
        
        residual_u = du_t +u_pred*du_x +v_pred*dv_y - self.nu*(d2u_x+d2u_y)
        residual_v = dv_t +v_pred*dv_x +v_pred*du_y - self.nu*(d2v_x+d2v_y)

        
        mse=nn.MSELoss()
        mse_u = mse(residual_u, torch.zeros_like(residual_u))
        mse_v = mse(residual_v, torch.zeros_like(residual_v))

        return mse_u, mse_v

    def compute_initial_condition(self, model, x_0, y_0):
        t_0 = torch.zeros_like(x_0)
        u_0, v_0 = torch.split(model(x_0, y_0, t_0), 1, dim=1) # de taille (N, 1), qu'on sépare en 2 (N, 1) contient toute les position pour t=0
        u_target = torch.sin(torch.pi*x_0)*torch.sin(torch.pi*y_0).to(device) # solution analytique
        v_target = torch.sin(torch.pi*x_0)*torch.sin(torch.pi*y_0).to(device) # solution analytique
        # u_target = 10*torch.exp(-((x_0 - 0.5) ** 2 + (y_0 - 0.5) ** 2) / (2 * 0.1 ** 2)).to(device)    # Gaussienne centrée en (0.5, 0.5) avec écart-type 0.1, sur le gpu car la mse s'y fait
        mse = nn.MSELoss()
        mse_u = mse(u_0, u_target)
        mse_v = mse(v_0, v_target)
        return mse_u + mse_v # on calcule la loss (valeur sous forme de tenseur direct)

    def compute_boundary_loss(self, model, bounds:dict, t):# bounds contient {i:(les x, u_target)} (en float) pour chaque point de bordure ou i correspond a la dimension : ex x: sur [x, 0] ca vaut u_target
        mse = nn.MSELoss()
        su=torch.tensor(0.0, device=device) # pour pouvoir sommer sur le gpu
        sv = torch.tensor(0.0, device=device)
        for k, v in bounds.items():
            if k=='x':
                x_bs = v[0]# on récupère les x
                y_bs = torch.zeros_like(x_bs)
                u_b, v_b = torch.split(model(x_bs, y_bs, t), 1, dim=1)
                u_target, v_target = v[1][0], v[1][1] # un couple qui contient toutes les valeurs de target
                u_ts = (torch.ones_like(t)*u_target)# pareil pour avoir la forme de t sur la target
                v_ts = torch.ones_like(t)*v_target
                su+=mse(u_b, u_ts)
                sv+=mse(v_b, v_ts)
            if k=='y':
                y_bs = v[0]# on récupère les y
                x_bs = torch.zeros_like(y_bs)
                u_b, v_b = torch.split(model(x_bs, y_bs, t), 1, dim=1)
                u_target, v_target = v[1][0], v[1][1] # un couple qui contient toutes les valeurs de target
                u_ts = (torch.ones_like(t)*u_target)# pareil pour avoir la forme de t sur la target
                v_ts = torch.ones_like(t)*v_target
                su+=mse(u_b, u_ts)
                sv+=mse(v_b, v_ts)
        return su + sv

    def compute_total_loss(self, model, x, y, t, bounds, a=1, b=1, c=1):
        pde_u, pde_v = self.compute_pde_residual(model, x, y, t)
        ic = self.compute_initial_condition(model, x, y)
        bl = self.compute_boundary_loss(model, bounds, t)
        return 10*(pde_u + pde_v) + 100*ic + bl

class Solver:
    def __init__(self, model, domain, physics):
        self.losses = []
        self.model = model

        self.domain = domain
        x, y, t = domain.sample_2D_training()
        self.x, self.y, self.t = x.to(device), y.to(device), t.to(device)# on travaille sur le gpu (mps)
        self.bounds = {'x':(self.x, (torch.zeros_like(self.x), torch.zeros_like(self.x))), 'y':(self.y, (torch.zeros_like(self.x), torch.zeros_like(self.y)))}
        self.physics = physics
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self, num_epochs, bounds={0.0:0.0, 1.0:0.0}):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            #x_gpu, y_gpu, t_gpu = self.x.to(device), self.y.to(device), self.t.to(device)
            # u_pred = self.u(x_gpu, y_gpu, t_gpu)
            loss = self.physics.compute_total_loss(self.model, self.x, self.y, self.t, self.bounds)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
            loss.backward()

            self.optimizer.step()
            self.losses.append(loss.item())

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
    physics = Physics(0.001)
    domain = Domain(t_max=2)
    solver = Solver(model, domain, physics)
    solver.train(2000, solver.bounds)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        u, v = torch.split(model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device)), 1, dim=1)
    arr_u = u.squeeze().detach().cpu().numpy()
    arr_v = u.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_u, domain)

    plot_w_slider_2d(arr_v, domain)
