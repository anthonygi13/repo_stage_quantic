import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

folder = "data1/"
all_a = np.loadtxt(folder + "all_a")
all_b = np.loadtxt(folder + "all_b")
all_d = np.loadtxt(folder + "all_d")
all_l = np.loadtxt(folder + "all_l", dtype=int)
all_m = np.loadtxt(folder + "all_m", dtype=int)
all_n = np.loadtxt(folder + "all_n", dtype=int)
list_Delta_i = np.loadtxt(folder + "list_Delta_i")
list_gi = np.loadtxt(folder + "list_gi")
list_omega_i = np.loadtxt(folder + "list_omega_i")
list_Q = np.loadtxt(folder + "list_Q")
list_Qi = np.loadtxt(folder + "list_Qi")

a_list = np.unique(all_a)
b_list = np.unique(all_b)
d_list = np.unique(all_d)

list_dominant_Delta = np.amin(list_Delta_i, axis=1)

print(list_gi[(all_n==0)*(all_m==1)*(all_l==1)][(all_b==0.07)*(all_d==0.03)*(all_a==0.02)])
print(list_gi[(all_n==0)*(all_m==1)*(all_l==1)][(all_b==0.07)*(all_d==0.02)*(all_a==0.02)])

print(list_gi[(all_b==0.07)*(all_d==0.03)*(all_a==0.02)])
print(list_gi[(all_b==0.07)*(all_d==0.02)*(all_a==0.02)])

print(np.amin(list_gi[(all_b==0.07)*(all_d==0.03)*(all_a==0.02)] - list_gi[(all_b==0.07)*(all_d==0.02)*(all_a==0.02)]))


fontsize = 7
plt.rcParams['font.size'] = str(fontsize)

fig1, fig2 = plt.figure(), plt.figure()
axes1, axes2 = [], []
for i, b in enumerate(b_list):
    new_all_a = all_a[all_b == b]
    new_all_d = all_d[all_b == b]

    new_list_Q = list_Q[all_b == b]

    axes1 += [fig1.add_subplot(int(np.sqrt(b_list.size))+1, int(np.sqrt(b_list.size))+1, i+1)]  # f_cav
    grid_a, grid_d = np.meshgrid(a_list, d_list)
    grid = griddata((new_all_a, new_all_d), new_list_Q, (grid_a, grid_d), fill_value=np.nan)
    im1 = axes1[-1].imshow(np.flip(grid, axis=0), extent=(a_list[0]-(a_list[1]-a_list[0])/2, a_list[-1]+(a_list[1]-a_list[0])/2,
                                                    d_list[0]-(d_list[1]-d_list[0])/2, d_list[-1]+(d_list[1]-d_list[0])/2), cmap='Reds', vmin=np.amin(list_Q), vmax=np.amax(list_Q))
    axes1[-1].set_xticks(np.arange(a_list[0], a_list[-1]+(a_list[1]-a_list[0]), (a_list[1]-a_list[0])*2))
    axes1[-1].set_yticks(np.arange(a_list[0], a_list[-1]+(a_list[1]-a_list[0]), (a_list[1]-a_list[0])*2))
    axes1[-1].set_xlabel("a [m]", fontsize=fontsize)
    axes1[-1].set_ylabel("d [m]", fontsize=fontsize)
    axes1[-1].set_title("Q for b={}".format(b))
    cbar1 = fig1.colorbar(im1, ax=axes1[-1])
    cbar1.ax.set_ylabel('Q', rotation=270, labelpad=30)

    new_list_dominant_Delta = list_dominant_Delta[all_b == b]

    axes2 += [fig2.add_subplot(int(np.sqrt(b_list.size)) + 1, int(np.sqrt(b_list.size)) + 1, i + 1)]  # f_cav
    grid_a, grid_d = np.meshgrid(a_list, d_list)
    grid = griddata((new_all_a, new_all_d), new_list_dominant_Delta, (grid_a, grid_d), fill_value=np.nan)
    im2 = axes2[-1].imshow(np.flip(grid, axis=0),
                           extent=(a_list[0] - (a_list[1] - a_list[0]) / 2, a_list[-1] + (a_list[1] - a_list[0]) / 2,
                                   d_list[0] - (d_list[1] - d_list[0]) / 2, d_list[-1] + (d_list[1] - d_list[0]) / 2),
                           cmap='Reds', vmin=np.amin(list_dominant_Delta), vmax=np.amax(list_dominant_Delta))
    axes2[-1].set_xticks(np.arange(a_list[0], a_list[-1] + (a_list[1] - a_list[0]), (a_list[1] - a_list[0]) * 2))
    axes2[-1].set_yticks(np.arange(a_list[0], a_list[-1] + (a_list[1] - a_list[0]), (a_list[1] - a_list[0]) * 2))
    axes2[-1].set_xlabel("a [m]", fontsize=fontsize)
    axes2[-1].set_ylabel("d [m]", fontsize=fontsize)
    axes2[-1].set_title("Q for b={}".format(b))
    cbar2 = fig2.colorbar(im2, ax=axes2[-1])
    cbar2.ax.set_ylabel('Q', rotation=270, labelpad=30)

plt.show()

