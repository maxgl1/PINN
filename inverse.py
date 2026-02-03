import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import qmc
import numpy as np
from main import HeatPINN, Domain


class Physics:
    def __init__(self, sensor):
         # on defini alpha comme un paramètre
         # pour garantir que alpha de l'équation ne devienne pas négatif
        self.alpha_parameter = nn.Parameter(torch.tensor([1.0], device=device)) # important de définir alpha dans la méthode et pas dans un init

        self.sensor= sensor

    
    def compute_data_loss(self, u):
        x_sens, y_sens, t_sens = self.sensor.record()
        u_real = self.sensor.analytoc_sol()
        u_pred = u(x_sens, y_sens, t_sens)
        mse = nn.MSELoss()
        return mse(u_real, u_pred)
    
    def compute_pde_residual(self, u, x, y, t):
        alpha_physic = torch.exp(self.alpha_parameter)# important de calculer ce alpha physique ici et pas dans le init car à chaque fois il faut le redéfinir (sinon on aurait défini que une seule fois sa valeur). ici on calcule, on ne stock pas
        u_pred = u(x, y, t)
        D_u = torch.autograd.grad(u_pred, [x, y, t], grad_outputs=torch.ones_like(u_pred), create_graph=True)#calcule les dérivées par rappoort a x et t (create_graph pour pouvoir accéder a la dérivée seconde), il respecte l'ordre donc la première valeur de 
        # Le gradient à l'index [0] de ton résultat sera exactement la dérivée  du/dx au point(x0, t0)
        # permet de faire le bon gradient en mettant le grad_outputs (histoire de jacoienne)
        du_x, du_y , du_t = D_u[0], D_u[1], D_u[2]
        d2u_x = torch.autograd.grad(du_x, [x], grad_outputs=torch.ones_like(du_x), create_graph=True)[0] # on choppe l'element du tuple
        d2u_y = torch.autograd.grad(du_y, [y], grad_outputs=torch.ones_like(du_y), create_graph=True)[0] # on choppe l'element du tuple
        residual = du_t - alpha_physic * (d2u_x + d2u_y)
        mse=nn.MSELoss()
        return mse(residual, torch.zeros_like(residual))

    def compute_initial_condition(self, u, x_0, y_0):
        t_0 = torch.zeros_like(x_0)
        u_0 = u(x_0, y_0, t_0) # de taille (N, 1), contient toute les position pour t=0
        u_target = torch.sin(torch.pi*x_0)*torch.sin(torch.pi*y_0).to(device) # solution analytique
        # u_target = 10*torch.exp(-((x_0 - 0.5) ** 2 + (y_0 - 0.5) ** 2) / (2 * 0.1 ** 2)).to(device)    # Gaussienne centrée en (0.5, 0.5) avec écart-type 0.1, sur le gpu car la mse s'y fait
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
        data = self.compute_data_loss(u)
        ic = self.compute_initial_condition(u, x, y)
        bl = self.compute_boundary_loss(u, bounds, t)
        return pde + 10*ic + 10*bl + 100*data
    
class Sensor:
    def __init__(self, loc, t, real_alpha): # loc de la forme :[[0.25, 0.25],[0.75, 0.25],[0.25, 0.75],[0.5, 0.5]]
        self.loc = loc
        self.t = t
        self.real_alpha = torch.tensor(real_alpha, device=device)

    def record(self):
        x_all = torch.tensor([], device=device)
        y_all = torch.tensor([], device=device)
        t_all = torch.tensor([], device=device)
        for pos in self.loc:
            x_i = torch.ones_like(self.t)*pos[0]
            y_i = torch.ones_like(self.t)*pos[1]
            x_all = torch.cat((x_all, x_i), dim=0)
            y_all = torch.cat((y_all, y_i), dim=0)
            t_all = torch.cat((t_all, self.t), dim=0)
        return x_all, y_all, t_all # on forme les points pour faire l'inférence
    
    def analytoc_sol(self):
        x_sens, y_sens, t_sens = self.record()
        time = torch.exp(-2*(torch.pi)**(2)*self.real_alpha*t_sens)
        space = torch.sin(torch.pi*x_sens)*torch.sin(torch.pi*y_sens)
        return time*space


        
class Solver:
    def __init__(self, u, domain, physics):
        self.losses = []
        self.alphas = []
        self.u = u

        self.domain = domain
        x, y, t = domain.sample_2D_training()
        self.x, self.y, self.t = x.to(device), y.to(device), t.to(device)# on travaille sur le gpu (mps)
        self.bounds = {'x':(self.x, torch.zeros_like(self.x)), 'y':(self.y, torch.zeros_like(self.y))}
        self.physics = physics
        params = list(self.u.parameters()) +[physics.alpha_parameter] # on ajoute alpha aux paramètres du modèle
        self.optimizer = torch.optim.Adam(params, lr=0.01)

        

    def train(self, num_epochs):

        for epochs in range(num_epochs):
            self.optimizer.zero_grad() # toujours reinitialiser pour éviter accumulation du gradient
            #x_gpu, y_gpu, t_gpu = self.x.to(device), self.y.to(device), self.t.to(device)
            # u_pred = self.u(x_gpu, y_gpu, t_gpu)
            loss = self.physics.compute_total_loss(self.u, self.x, self.y, self.t, self.bounds)# calcul de la loss (optimisable car on rentre dedans le modèle et pas l'inférence donc on recalule plusieurs fois les memes choses)
            loss.backward()

            self.optimizer.step()

            self.losses.append(loss.item())
            self.alphas.append(torch.exp(physics.alpha_parameter).item())
            if epochs%20==0:
                print(f"epoch : {epochs}, loss = {loss.item()}, alpha = {physics.alpha_parameter.item()}")# on affiche le alpha mais c'est son exponentielle qu'il faut prendre (car c'est son exponentielle qu'on utilise dans la pde)



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

def plot_list(list, name):
    plt.plot(list)
    plt.title(f"Évolution de la {name} au cours des itérations")
    plt.show()
if __name__ == '__main__':
    torch.manual_seed(3)
    #train
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Calculs effectués sur : {device}")
    model = HeatPINN().to(device) # calculs sur le gpu
    domain = Domain(t_max=0.4)
    domain.sample_2D_training()# nécéssaire car le self.t défini dans la méthode
    sensor = Sensor([[0.25, 0.25],[0.75, 0.25],[0.25, 0.75],[0.5, 0.5]], domain.t.to(device), 0.5)# position des capteurs
    physics = Physics(sensor)
    solver = Solver(model, domain, physics)
    solver.train(4000)

    #test
    domain.create_grid_testing()
    with torch.no_grad(): # pas besoin de sauvegar
        y = model(domain.x_grid.to(device), domain.y_grid.to(device), domain.t_grid.to(device))
    arr_y = y.squeeze().detach().cpu().numpy()
    
    plot_w_slider_2d(arr_y, domain)
    plot_list(solver.losses, 'loss')
    plot_list(solver.alphas, 'alpha')
