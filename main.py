import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

class HeatPINN(nn.Module):# on fait hérier de nn.Module pour que pytorch gère ensuite seul et sache entre autre quel poids mettre à jour, tout les parametres du modèle seront dans self.parameters()
    def __init__(self):
        super().__init__()# on initialise la classe parente
        self.net = nn.ModuleList([nn.Linear(2, 20)]+[nn.Linear(20, 20) for _ in range(4)]+[nn.Linear(20, 1)]) # on définit juste des blocs qu'on pourra utiliser ou non (différent de sequential qui fait lui meme les enchainements)
    
    def forward(self, x, t):
        u = torch.cat([x, t], dim=1) # permet de passer de vecteurs colonnes (N, 1) en un seul (N, 2) (dim=1 signifie qu'on colle ligne par ligne et pas l'un au dessus de l'autre)
        
        for module in self.net[:-1]:
            u=module(u)
            u=torch.tanh(u) # fonction d'activation (valeur dans [-1, 1])
        u=self.net[-1](u) # le dernier pas de tanh pour ne pas avoir que des valeurs bornées"""
        return u[:, 0:1] # de taille (N, 1)

class Domain:
    def __init__(self):
        self.x_collocation = torch.linspace(0, 1, 50, requires_grad=True)# on initialise la grille (N, 1)
        self.t_collocation = torch.linspace(0, 2, 50, requires_grad=True) #on souhaite pouvoir suivre ce qu'il se passe pour pouvoir règle de la chaie
        
        self.X, self.T = torch.meshgrid(self.x_collocation, self.t_collocation, indexing='ij') # forme les 2 grilles
        
        self.x_grid = self.X.reshape(-1, 1)
        self.t_grid = self.T.reshape(-1, 1)


class Physics:
    def __init__(self, alpha):
        self.alpha=alpha

    def compute_pde_residual(self, u, x, t):
        u_pred = u(x, t)
        D_u = torch.autograd.grad(u_pred, [x, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        # Le gradient à l'index [0] de ton résultat sera exactement la dérivée  du/dx au point(x0, t0)
        # permet de faire le bon gradient en mettant le grad_outputs (histoire de jacoienne)
        du_x, du_t = D_u[0], D_u[1]
        d2u_x = torch.autograd.grad(du_x, [x], grad_outputs=torch.ones_like(du_x), create_graph=True)[0] # on choppe l'element du tuple
        residual = du_t - self.alpha * d2u_x
        mse=nn.MSELoss()
        return mse(residual, torch.zeros_like(residual))

    def compute_initial_condition(self, u, x_0):
        t_0 = torch.zeros(x_0.shape[0], 1)
        u_0 = u(x_0, t_0) # de taille (N, 1), contient toute les position pour t=0
        u_target = torch.sin(torch.pi * x_0)
        mse = nn.MSELoss()
        return mse(u_0, u_target) # on calcule la loss (valeur sous forme de tenseur direct)

    def compute_boundary_loss(self, u, bounds:dict, t):# bounds contient {xb:u_target} (en float) pour chaque point de bordure
        mse = nn.MSELoss()
        s=torch.tensor(0.0)
        for x_b, u_target in bounds.items():
            x_bs = torch.ones_like(t)*x_b# on forme la liste des x identiques mais de la meme taille que t (pas besoin de mettre x_b en tenseur car torch le fais seul)
            u_b = u(x_bs, t)
            u_ts = torch.ones(t.shape[0], 1)*u_target# pareil pour avoir la forme de t sur la target
            s+=mse(u_b, u_ts)
        return s

    def compute_total_loss(self, u, x, t, bounds, a=1, b=1, c=1):
        pde = self.compute_pde_residual(u, x, t)
        ic = self.compute_initial_condition(u, x)
        bl = self.compute_boundary_loss(u, bounds, t)
        return pde + ic + bl

class Solver:
    def __init__(self, u, domain, physics):
        self.u = u

        self.domain = domain
        self.x = domain.x_grid
        self.t = domain.t_grid

        self.physics = physics

        self.optimizer = torch.optim.Adam(self.u.parameters(), lr=0.01)

    def train(self, num_epochs, bounds={0.0:0.0, 1.0:0.0}):
        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            u_pred = self.u(self.x, self.t)
            loss = self.physics.compute_total_loss(self.u, self.x, self.t, bounds)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
            print(loss)
            loss.backward()
            
            self.optimizer.step()

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


if __name__ == '__main__':
    model = HeatPINN()
    physics = Physics(0.5)
    domain = Domain()
    solver = Solver(model, domain, physics)
    solver.train(300)
    
    y = model(domain.x_grid, domain.t_grid)
    arr_y = convert_tensor_to_array(y)
    
    plot_w_slider(arr_y, domain)
    

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
