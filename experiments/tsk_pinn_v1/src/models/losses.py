import torch


def burgers_residual(model, x, nu):
    x = x.clone().detach().requires_grad_(True)
    u, _ = model(x)
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_x = grad[:, 0:1]
    u_t = grad[:, 1:2]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0][:, 0:1]
    return u_t + u * u_x - nu * u_xx


def mse(a, b):
    return ((a - b) ** 2).mean()
