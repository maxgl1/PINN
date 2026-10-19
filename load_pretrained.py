from heat import HeatPINN
from burger import BurgerPINN
from navier import NavierPINN, plot_w_slider_2d
import torch
from navier_optimised import animate_pinn_final, Domain

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def void(u, v, p, x, y):
    du_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    return du_x - dv_y

def inference(equation, t_val=4):
    domain = Domain(t_min=0, t_max=20, x_min=1, x_max=8, y_min=-2, y_max=2)
    x, y, t = domain.create_grid_testing()
    x = x.to(device).requires_grad_(True)
    y = y.to(device).requires_grad_(True)
    t = t.to(device)

    if equation == 'navier':
        model = NavierPINN().to(device)
        model.load_state_dict(torch.load("pretrained/PINN_navier_cylinder0.0005.pth", map_location=device))
        u, v, p = torch.split(model(x, y, t), 1, dim=1)

        w = void(u, v, p, x, y).detach().cpu()
        with torch.no_grad():
            u, v, p = torch.split(model(x, y, t), 1, dim=1)
        u, v, p = u.cpu(), v.cpu(), p.cpu()
        plot_w_slider_2d(w, domain, "w", False)
        plot_w_slider_2d(u, domain, "u", False)
        plot_w_slider_2d(v, domain, "v", False)
        plot_w_slider_2d(p, domain, "p", False)

    if equation == 'heat':
        model = HeatPINN().to(device)
        model.load_state_dict(torch.load("pretrained/PINN_heat.pth", map_location=device))
        with torch.no_grad():
            T = model(x, y, t).cpu()
        plot_w_slider_2d(T, domain)

    if equation == "burger":
        model = BurgerPINN().to(device)
        model.load_state_dict(torch.load("pretrained/PINN_burger.pth", map_location=device))
        with torch.no_grad():
            u, v = torch.split(model(x, y, t), 1, dim=1)
        u, v = u.cpu(), v.cpu()
        plot_w_slider_2d(u, domain, "u")
        plot_w_slider_2d(v, domain, "v")

if __name__ == "__main__":
    inference("navier", 4)