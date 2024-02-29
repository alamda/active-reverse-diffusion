import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import torch
import seaborn as sns
import itertools
from tqdm.auto import tqdm
import copy
import scipy.special as special
from sklearn.neighbors import KernelDensity


def do_diffusion(data, tsteps, T, dt):
    distributions, samples = [None], [data]
    xt = data
    for t in range(tsteps):
        xt = xt - dt*xt + \
            np.sqrt(2*T*dt)*torch.normal(torch.zeros_like(data),
                                         torch.ones_like(data))
        samples.append(xt)
    return samples


def compute_loss(forward_samples, T, t, dt, score_model):
    xt = forward_samples[t].type(torch.DoubleTensor)         # x(t)
    l = -(xt - forward_samples[0]*np.exp(-t*dt)) / \
        (T*(1-np.exp(-2*t*dt)))  # The actual score function
    scr = score_model(xt)
    loss = torch.mean((scr - l)**2)
    return loss, torch.mean(l**2), torch.mean(scr**2)


def passive_training(dataset, tsteps, T, dt, nrnodes=4, iterations=500):
    loss_history = []
    all_models = []
    forward_samples = do_diffusion(dataset, tsteps, T, dt)
    if len(forward_samples[0].shape) == 1:
        num_points = forward_samples[0].shape[0]

        forward_samples = [f.reshape((num_points, 1)) for f in forward_samples]
    bar = tqdm(range(1, tsteps))
    t = 1
    for e in bar:
        score_model = torch.nn.Sequential(
            torch.nn.Linear(1, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, 1)
        ).double()
        optim = torch.optim.AdamW(itertools.chain(
            score_model.parameters()), lr=1e-2, weight_decay=1e-8,)
        loss = 100
        for _ in range(iterations):
            optim.zero_grad()
            loss, l, scr = compute_loss(forward_samples, T, t, dt, score_model)
            loss.backward()
            optim.step()
        bar.set_description(f'Time:{t} Loss: {loss.item():.4f}')
        all_models.append(copy.deepcopy(score_model))
        loss_history.append(loss.item())
        t = t + 1
    return all_models


def sampling(N, all_models, T, dt, tsteps):
    x = T*torch.randn(N, 1).type(torch.DoubleTensor)
    samples = [x.detach()]
    for t in range(tsteps-2, 0, -1):
        F = all_models[t](x)
        # If using the total score function
        x = x + x*dt + 2*T*F*dt + np.sqrt(2*T*dt)*torch.randn(N, 1)
        samples.append(x.detach())
    return x.detach(), samples


def do_diffusion_active(data, tsteps, Tp, Ta, tau, dt):
    eta = torch.normal(torch.zeros_like(
        data), np.sqrt(Ta/tau)*torch.ones_like(data))
    samples = [data]
    eta_samples = [eta]
    xt = data
    for t in range(tsteps):
        xt = xt - dt*xt + dt*eta + \
            np.sqrt(2*Tp*dt)*torch.normal(torch.zeros_like(data),
                                          torch.ones_like(data))
        eta = eta - (1/tau)*dt*eta + (1/tau)*np.sqrt(2*Ta*dt) * \
            torch.normal(torch.zeros_like(eta), torch.ones_like(eta))
        samples.append(xt)
        eta_samples.append(eta)
    return samples, eta_samples


def M_11_12_22(Tp, Ta, tau, k, t):
    a = np.exp(-k*t)
    b = np.exp(-t/tau)
    Tx = Tp
    Ty = Ta/(tau*tau)
    w = (1/tau)
    M11 = (1/k)*Tx*(1-a*a) + (1/k)*Ty*(1/(w*(k+w)) + 4*a*b *
                                       k/((k+w)*(k-w)**2) - (k*b*b + w*a*a)/(w*(k-w)**2))
    M12 = (Ty/(w*(k*k - w*w))) * (k*(1-b*b) - w*(1 + b*b - 2*a*b))
    M22 = (Ty/w)*(1-b*b)
    return M11, M12, M22


def score_function_loss(forward_samples, forward_samples_eta, Tp, Ta, tau, k, t, dt, score_model_x, score_model_eta):
    a = np.exp(-k*t*dt)
    b = (np.exp(-t*dt/tau) - np.exp(-k*t*dt))/(k-(1/tau))
    c = np.exp(-t*dt/tau)
    M11, M12, M22 = M_11_12_22(Tp, Ta, tau, k, t*dt)
    det = M11*M22 - M12*M12
    x0 = forward_samples[0]
    eta0 = forward_samples_eta[0]
    x = forward_samples[t].type(torch.DoubleTensor)
    eta = forward_samples_eta[t].type(torch.DoubleTensor)
    Fx = (1/det)*(-M22*(x - a*x0 - b*eta0) + M12*(eta - c*eta0))
    Feta = (1/det)*(-M11*(eta - c*eta0) + M12*(x - a*x0 - b*eta0))

    if len(Fx.shape) == 1:
        Fx = Fx.reshape((Fx.shape[0], 1))
    if len(Feta.shape) == 1:
        Feta = Feta.reshape((Feta.shape[0], 1))
    if len(x.shape) == 1:
        x = x.reshape((x.shape[0], 1))
    if len(eta.shape) == 1:
        eta = eta.reshape((eta.shape[0], 1))

    F = torch.cat((Fx, Feta), dim=1)
    xin = torch.cat((x, eta), dim=1)
    scr_x = score_model_x(xin)
    scr_eta = score_model_eta(xin)
    loss_x = torch.mean((scr_x - Fx)**2)
    loss_eta = torch.mean((scr_eta - Feta)**2)
    return loss_x, loss_eta, torch.mean(Fx**2), torch.mean(Feta**2), torch.mean(scr_x**2), torch.mean(scr_eta**2)


def active_training(dataset, tsteps, Tp, Ta, tau, k, dt, nrnodes=4, iterations=500):
    loss_history = []
    all_models_x = []
    all_models_eta = []
    forward_samples, forward_samples_eta = do_diffusion_active(
        dataset, tsteps, Tp, Ta, tau, dt)

    if len(forward_samples[0]) == 1:
        num_points = forward_samples[0].shape[0]
        forward_samples = [f.reshape((num_points, 1)) for f in forward_samples]

    if len(forward_samples_eta[0]) == 1:
        num_points = forward_samples_eta[0].shape[0]
        forward_samples_eta = [f.reshape((num_points, 1))
                               for f in forward_samples_eta]
    t = 1
    bar = tqdm(range(1, tsteps))
    for e in bar:
        score_model_x = torch.nn.Sequential(
            torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, 1)
        ).double()
        score_model_eta = torch.nn.Sequential(
            torch.nn.Linear(2, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, nrnodes), torch.nn.Tanh(),
            torch.nn.Linear(nrnodes, 1)
        ).double()
        optim_x = torch.optim.AdamW(itertools.chain(
            score_model_x.parameters()), lr=1e-2, weight_decay=1e-8,)
        optim_eta = torch.optim.AdamW(itertools.chain(
            score_model_eta.parameters()), lr=1e-2, weight_decay=1e-8,)
        loss = 100
        for _ in range(iterations):
            optim_x.zero_grad()
            optim_eta.zero_grad()
            loss_x, loss_eta, loss_Fx, loss_Feta, loss_scr_x, loss_scr_eta = score_function_loss(
                forward_samples, forward_samples_eta, Tp, Ta, tau, k, t, dt, score_model_x, score_model_eta)
            loss_x.backward()
            loss_eta.backward()
            optim_x.step()
            optim_eta.step()
        bar.set_description(
            f'Time:{t} Loss: {loss_x.item():.4f} Fx: {loss_Fx.item():.4f} scr_x: {loss_scr_x.item():.4f}')
        all_models_x.append(copy.deepcopy(score_model_x))
        all_models_eta.append(copy.deepcopy(score_model_eta))
        loss_history.append(loss_x.item())
        t = t + 1
    return all_models_x, all_models_eta


def sampling_active(N, all_models_x, all_models_eta, Tp, Ta, tau, k, dt, tsteps):
    x = np.sqrt(Tp/k + (Ta/(k*k*tau+k))) * \
        torch.randn([N, 1]).type(torch.DoubleTensor)
    eta = np.sqrt(Ta/tau)*torch.randn([N, 1]).type(torch.DoubleTensor)
    samples_x = [x.detach()]
    samples_eta = [eta.detach()]
    for t in range(tsteps-2, 0, -1):
        xin = torch.cat((x, eta), dim=1)
        Fx = all_models_x[t](xin)
        Feta = all_models_eta[t](xin)
        x = x + dt*(x-eta) + 2*Tp*Fx*dt + np.sqrt(2*Tp*dt)*torch.randn(x.shape)
        eta = eta + dt*eta/tau + (2*Ta/(tau*tau))*Feta*dt + \
            (1/tau)*np.sqrt(2*Ta*dt)*torch.randn(eta.shape)
        samples_x.append(x.detach())
        samples_eta.append(eta.detach())
    return x.detach(), eta.detach(), samples_x, samples_eta
