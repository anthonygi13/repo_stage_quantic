import numpy as np
import numdifftools as nd
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def c():
    return 299792458

def hbar():
    return 6.62607015e-34/2/np.pi

def epsilon0():
    return 8.85418782e-12

def mu0():
    return 1.25663706212e-6

def Rs(omega):
    sigma = 5.813e7  # copper
    return np.sqrt(omega*mu0()/2/sigma)

def epsilon_eff():
    return (10+1)/2

def get_omega(k):
    return c()*np.linalg.norm(k)

def get_k(a, b, d, n, m, l):
    return np.array([np.pi*n/a, np.pi*m/b, np.pi*l/d])

def get_polarisation(k):
    if np.count_nonzero(k) < 3:
        res = np.ones(k.shape)
        res[k != 0] = 0
        return res, None
    else:
        res1 = np.array([-k[1], k[0], 0])
        res1 /= np.linalg.norm(res1)
        res2 = np.cross(k/np.linalg.norm(k), res1)
        return res1, res2

def get_E(a, b, d, n, m, l):  # E quantique
    V = a*b*d
    k = get_k(a, b, d, n, m, l)
    polar1, polar2 = get_polarisation(k)
    if polar2 is None:
        N = np.sqrt(4/V)*np.sqrt(hbar()*get_omega(k)/2/epsilon0())
    else:
        N = np.sqrt(8/V)*np.sqrt(hbar()*get_omega(k)/2/epsilon0())
    Ex1 = lambda x: N*polar1[0] * np.cos(k[0]*x[0])*np.sin(k[1]*x[1])*np.sin(k[2]*x[2])
    Ey1 = lambda x: N*polar1[1] * np.sin(k[0]*x[0])*np.cos(k[1]*x[1])*np.sin(k[2]*x[2])
    Ez1 = lambda x: N*polar1[2] * np.sin(k[0]*x[0])*np.sin(k[1]*x[1])*np.cos(k[2]*x[2])
    if polar2 is None:
        return lambda x: np.array([Ex1(x), Ey1(x), Ez1(x)]), None
    else:
        Ex2 = lambda x: N * polar2[0] * np.cos(k[0] * x[0]) * np.sin(k[1] * x[1]) * np.sin(k[2] * x[2])
        Ey2 = lambda x: N * polar2[1] * np.sin(k[0] * x[0]) * np.cos(k[1] * x[1]) * np.sin(k[2] * x[2])
        Ez2 = lambda x: N * polar2[2] * np.sin(k[0] * x[0]) * np.sin(k[1] * x[1]) * np.cos(k[2] * x[2])
    return lambda x: np.array([Ex1(x), Ey1(x), Ez1(x)]), lambda x: np.array([Ex2(x), Ey2(x), Ez2(x)])

def get_omega_l(L):
    return 2*np.pi/(L*2)*c()/np.sqrt(epsilon_eff())

def get_charge(L, zl):
    # approximated
    C = 1
    return 1/L * np.sqrt(hbar()*get_omega_l(L)*L*C) * np.cos(zl*np.pi/L)

def get_gi(a, b, d, E, L, e):
    func = lambda zl: E(np.array([a/2, b/2, d/2-L/4+zl]))[0]*get_charge(L, zl)*e
    return integrate.quad(func, 0, L/2, epsabs=0)[0]

def get_pi(gi, Delta_i):
    return (gi/Delta_i)**2

def curl(f):
    jac = nd.Jacobian(f)
    return lambda x: np.array([jac(x)[2,1]-jac(x)[1,2],jac(x)[0,2]-jac(x)[2,0],jac(x)[1,0]-jac(x)[0,1]])

def get_B_old(E, k):  # B quantique
    return lambda x: curl(E)(x) / c() / np.linalg.norm(k)

def get_B(a, b, d, n, m, l):  # B quantique
    V = a*b*d
    k = get_k(a, b, d, n, m, l)
    polar1, polar2 = get_polarisation(k)
    if polar2 is None:
        N = np.sqrt(4/V)*np.sqrt(hbar()*get_omega(k)/2*mu0())/np.linalg.norm(k)
    else:
        N = np.sqrt(8/V)*np.sqrt(hbar()*get_omega(k)/2*mu0())/np.linalg.norm(k)
    Bx1 = lambda x: N * (polar1[2]*k[1] - polar1[1]*k[2]) * np.sin(k[0]*x[0])*np.cos(k[1]*x[1])*np.cos(k[2]*x[2])
    By1 = lambda x: N * (polar1[0]*k[2] - polar1[2]*k[0]) * np.cos(k[0]*x[0])*np.sin(k[1]*x[1])*np.cos(k[2]*x[2])
    Bz1 = lambda x: N * (polar1[1]*k[0] - polar1[0]*k[1]) * np.cos(k[0]*x[0])*np.cos(k[1]*x[1])*np.sin(k[2]*x[2])
    if polar2 is None:
        return lambda x: np.array([Bx1(x), By1(x), Bz1(x)]), None
    else:
        Bx2 = lambda x: N * (polar2[2] * k[1] - polar2[1] * k[2]) * np.sin(k[0] * x[0]) * np.cos(k[1] * x[1]) * np.cos(
            k[2] * x[2])
        By2 = lambda x: N * (polar2[0] * k[2] - polar2[2] * k[0]) * np.cos(k[0] * x[0]) * np.sin(k[1] * x[1]) * np.cos(
            k[2] * x[2])
        Bz2 = lambda x: N * (polar2[1] * k[0] - polar2[0] * k[1]) * np.cos(k[0] * x[0]) * np.cos(k[1] * x[1]) * np.sin(
            k[2] * x[2])
    return lambda x: np.array([Bx1(x), By1(x), Bz1(x)]), lambda x: np.array([Bx2(x), By2(x), Bz2(x)])


def get_Qi(a, b, d, E, B, omega):
    func = lambda x, y, z: epsilon0()/4*np.sum(E(np.array([x, y, z]))**2)
    We = integrate.tplquad(func, 0, d, 0, b, 0, a, epsabs=0)[0]
    #print("We", We)

    Pc = 0
    funcx = lambda y, z: Rs(omega) / 2 * (np.sum((B(np.array([0, y, z]))[1:] / mu0())**2) + np.sum((B(np.array([a, y, z]))[1:]/mu0())**2))
    Pc += integrate.dblquad(funcx, 0, d, 0, b, epsabs=0)[0]

    funcy = lambda x, z: Rs(omega)/2 * (np.sum((B(np.array([x, 0, z]))[[0, 2]]/mu0())**2) + np.sum((B(np.array([x, b, z]))[[0, 2]]/mu0())**2))
    Pc += integrate.dblquad(funcy, 0, d, 0, a, epsabs=0)[0]

    funcz = lambda x, y: Rs(omega)/2 * (np.sum((B(np.array([x, y, 0]))[:-1]/mu0())**2) + np.sum((B(np.array([x, y, d]))[:-1]/mu0())**2))
    Pc += integrate.dblquad(funcz, 0, b, 0, a, epsabs=0)[0]

    #print("Pc", Pc)
    return 2*omega*We/Pc

def check_Q(a, b, d, k, omega):
    return (np.linalg.norm(k)*a*d)**3*b*np.sqrt(mu0()/epsilon0())/(2*np.pi**2*Rs(omega)*(2*a**3*b+2*b*d**3+a**3*d+a*d**3))

"""
a, b, d = 1*1e-2, 2*1e-2, 3*1e-2
n, m, l = 1, 0, 1
E1, E2 = get_E(a, b, d, n, m, l)
B1, B2 = get_B(a, b, d, n, m, l)
k = get_k(a, b, d, n, m, l)
omega = get_omega(k)
Q = get_Qi(a, b, d, E1, B1, omega)
Q2 = check_Q(a, b, d, k, omega)
E0 = E1([a/2, b/2, d/2])[1]
print("check We", epsilon0()*a*b*d/16*E0**2)
print("check Pc", Rs(omega)*E0**2*(2*np.pi/np.linalg.norm(k))**2/(8*mu0()/epsilon0())
      * (a*b/d**2 + b*d/a**2 + a/2/d + d/2/a))

#print(B1(np.array([0.4, 0.4, 0.4])*1e-2))
#print(get_B_old(E1, k)(np.array([0.4, 0.4, 0.4])*1e-2))

print(E0/(np.linalg.norm(k)*np.sqrt(mu0()/epsilon0())/np.sqrt(np.linalg.norm(k)**2-(np.pi/a)**2)), B1(np.array([a/2, 0, 0]))[0]/mu0())
print(np.pi*E0/(np.linalg.norm(k)*np.sqrt(mu0()/epsilon0())*a), B1(np.array([0, 0, d/2]))[2]/mu0())

print(Q)
print(Q2)
"""

folder = "data1/"

L = 1.28e-2
e = 45e-6
omega_l = get_omega_l(L)

a_list = np.arange(1, 5, 1) * 1e-2
b_list = np.arange(4, 8, 1) * 1e-2
d_list = np.arange(1, 5, 1) * 1e-2
n_list = np.arange(0, 4)
m_list = np.arange(0, 4)
l_list = np.arange(0, 4)

all_n = []
all_m = []
all_l = []
all_a = []
all_b = []
all_d = []
list_omega_i = []
list_Delta_i = []
list_Qi = []
list_gi = []
i = 0
for a in a_list:
    for b in b_list:
        for d in d_list:
            i += 1
            print("{}/{}".format(i, a_list.size*b_list.size*d_list.size))
            all_a += [a]
            all_b += [b]
            all_d += [d]
            all_n += [[]]
            all_m += [[]]
            all_l += [[]]
            list_omega_i += [[]]
            list_Qi += [[]]
            list_gi += [[]]
            list_Delta_i += [[]]
            for n in n_list:
                for m in m_list:
                    if n == 0 and m == 0:
                        continue
                    for l in l_list:
                        if n == 0 and l == 0 or m == 0 and l == 0:
                            continue

                        all_n[-1] += [n]
                        all_m[-1] += [m]
                        all_l[-1] += [l]

                        k = get_k(a, b, d, n, m, l)
                        omega = get_omega(k)
                        E1, E2 = get_E(a, b, d, n, m, l)
                        B1, B2 = get_B(a, b, d, n, m, l)
                        list_omega_i[-1] += [omega]
                        list_Delta_i[-1] += [omega - omega_l]
                        list_Qi[-1] += [get_Qi(a, b, d, E1, B1, omega)]
                        list_gi[-1] += [get_gi(a, b, d, E1, L, e)]

                        if E2 is not None:
                            all_n[-1] += [n]
                            all_m[-1] += [m]
                            all_l[-1] += [l]
                            list_omega_i[-1] += [omega]
                            list_Delta_i[-1] += [omega - omega_l]
                            list_Qi[-1] += [get_Qi(a, b, d, E2, B2, omega)]
                            list_gi[-1] += [get_gi(a, b, d, E2, L, e)]

all_n = np.array(all_n)
all_m = np.array(all_m)
all_l = np.array(all_l)
all_a = np.array(all_a)
all_b = np.array(all_b)
all_d = np.array(all_d)
list_Qi = np.array(list_Qi)
list_gi = np.array(list_gi)
list_omega_i = np.array(list_omega_i)
list_Delta_i = np.array(list_Delta_i)
list_Q = omega_l/np.sum((list_gi/list_Delta_i)**2 * list_omega_i/list_Qi, axis=1)

np.savetxt(folder+"all_n", all_n)
np.savetxt(folder+"all_m", all_m)
np.savetxt(folder+"all_l", all_l)
np.savetxt(folder+"all_a", all_a)
np.savetxt(folder+"all_b", all_b)
np.savetxt(folder+"all_d", all_d)
np.savetxt(folder+"list_Qi", list_Qi)
np.savetxt(folder+"list_gi", list_gi)
np.savetxt(folder+"list_omega_i", list_omega_i)
np.savetxt(folder+"list_Delta_i", list_Delta_i)
np.savetxt(folder+"list_Q", list_Q)

fontsize = 7
plt.rcParams['font.size'] = str(fontsize)

fig = plt.figure()
axes = []
for i, b in enumerate(b_list):
    new_list_Q = list_Q[all_b == b]
    new_all_a = all_a[all_b == b]
    new_all_d = all_d[all_b == b]

    axes += [fig.add_subplot(int(np.sqrt(b_list.size))+1, int(np.sqrt(b_list.size))+1, i+1)]  # f_cav
    grid_a, grid_d = np.meshgrid(a_list, d_list)
    grid = griddata((new_all_a, new_all_d), new_list_Q, (grid_a, grid_d), fill_value=np.nan)
    im1 = axes[-1].imshow(np.flip(grid, axis=0), extent=(a_list[0]-(a_list[1]-a_list[0])/2, a_list[-1]+(a_list[1]-a_list[0])/2,
                                                    d_list[0]-(d_list[1]-d_list[0])/2, d_list[-1]+(d_list[1]-d_list[0])/2), cmap='Reds', vmin=np.amin(list_Q), vmax=np.amax(list_Q))
    axes[-1].set_xticks(np.arange(a_list[0], a_list[-1]+(a_list[1]-a_list[0]), (a_list[1]-a_list[0])*2))
    axes[-1].set_yticks(np.arange(a_list[0], a_list[-1]+(a_list[1]-a_list[0]), (a_list[1]-a_list[0])*2))
    axes[-1].set_xlabel("a [m]", fontsize=fontsize)
    axes[-1].set_ylabel("d [m]", fontsize=fontsize)
    axes[-1].set_title("Q for b={}".format(b))
    cbar1 = fig.colorbar(im1, ax=axes[-1])
    cbar1.ax.set_ylabel('Q', rotation=270, labelpad=30)

plt.show()
