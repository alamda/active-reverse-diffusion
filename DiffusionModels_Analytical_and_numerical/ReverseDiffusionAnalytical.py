import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
import scipy.interpolate as interpolate


def forward_process(x, T, tsteps, dt):
    xmat = np.zeros((len(x), tsteps+1))
    xmat[:, 0] = x
    # range starts from 0
    for t in range(tsteps):
        x = x - x*dt + np.sqrt(2*T*dt)*np.random.randn(len(x))
        xmat[:, t] = x
    return xmat


def score_function_passive_new_multiple(x, T, mulist, sigmalist, plist, t):
    a = np.exp(-t)
    Delta = T*(1-np.exp(-2*t))
    Fx_num = 0
    Fx_den = 0
    for i in range(len(mulist)):
        mu = mulist[i]
        sigma = sigmalist[i]
        p = plist[i]
        h = sigma*sigma
        Delta_eff = a*a*h + Delta
        z = np.exp((-(x-a*mu)**2)/(2*Delta_eff))
        Fx_num = Fx_num - p*np.sqrt(h)*np.power(Delta_eff, -1.5)*(x-a*mu)*z
        Fx_den = Fx_den + p*np.sqrt(h)*np.power(Delta_eff, -0.5)*z
    Fx = Fx_num/Fx_den
    return Fx


def reverse_process_passive_new_multiple(x, T, tsteps, dt, sigmalist, mulist, plist):
    xmat = np.zeros((len(x), tsteps+1))
    xmat[:, tsteps] = x
    for t in range(tsteps, 0, -1):
        Fx = score_function_passive_new_multiple(
            x, T, mulist, sigmalist, plist, t*dt)
        x = x + dt*x + 2*T*Fx*dt + np.sqrt(2*T*dt)*np.random.randn(len(x))
        xmat[:, t-1] = x
    return xmat


def forward_process_active(x, y, Tp, Ta, tau, tsteps, dt):
    xmat = np.zeros((len(x), tsteps+1))
    xmat[:, 0] = x
    ymat = np.zeros((len(y), tsteps+1))
    ymat[:, 0] = y
    for t in range(tsteps+1):
        x = x - x*dt + y*dt + np.sqrt(2*Tp*dt)*np.random.randn(len(x))
        y = y - (y/tau)*dt + (1/tau)*np.sqrt(2*Ta*dt)*np.random.randn(len(y))
        xmat[:, t] = x
        ymat[:, t] = y
    return xmat, ymat


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


def score_function_new_multiple(x, eta, Tp, Ta, tau, mulist, sigmalist, plist, k, t):
    a = np.exp(-k*t)
    b = (np.exp(-t/tau) - np.exp(-k*t))/(k-(1/tau))
    c = np.exp(-t/tau)
    g = Ta/tau
    M11, M12, M22 = M_11_12_22(Tp, Ta, tau, k, t)
    Fx_num = 0
    Fx_den = 0
    Feta_num = 0
    Feta_den = 0
    for i in range(len(mulist)):
        mu = mulist[i]
        sigma = sigmalist[i]
        p = plist[i]
        h = sigma*sigma
        K1 = c*c*g + M22
        K2 = b*c*g + M12
        K3 = b*b*g + a*a*h + M11
        Delta_eff = c*c*g*M11 - 2*b*c*g*M12 - M12*M12 + \
            b*b*g*M22 + M11*M22 + a*a*h*(c*c*g + M22)
        z = np.exp((-K1*(x-a*mu)**2 + 2*K2*(x-a*mu)
                   * eta - K3*eta**2)/(2*Delta_eff))
        Fx_num = Fx_num + p*h * \
            np.power(Delta_eff, -1.5)*(K2*eta - K1*(x-a*mu))*z
        Fx_den = Fx_den + p*h*np.power(Delta_eff, -0.5)*z
        Feta_num = Feta_num + p*h * \
            np.power(Delta_eff, -1.5)*(K2*(x-a*mu) - K3*eta)*z
        Feta_den = Feta_den + p*h*np.power(Delta_eff, -0.5)*z
    Fx = Fx_num/Fx_den
    Feta = Feta_num/Feta_den
    return Fx, Feta


def reverse_process_active_new_multiple(x, y, Tp, Ta, tau, tsteps, dt, sigmalist, mulist, plist, k):
    xmat = np.zeros((len(x), tsteps+1))
    ymat = np.zeros((len(y), tsteps+1))
    xmat[:, tsteps] = x
    ymat[:, tsteps] = y
    for t in range(tsteps, 0, -1):
        Fx, Fy = score_function_new_multiple(
            x, y, Tp, Ta, tau, mulist, sigmalist, plist, k, t*dt)
        x = x + dt*(x-y) + 2*Tp*Fx*dt + np.sqrt(2*Tp*dt) * \
            np.random.randn(len(x))
        y = y + dt*y/tau + (2*Ta/(tau*tau))*Fy*dt + (1/tau) * \
            np.sqrt(2*Ta*dt)*np.random.randn(len(y))
        xmat[:, t-1] = x
        ymat[:, t-1] = y
    return xmat, ymat
