import sys
sys.path.append('/home/bili/Lernen/PhaseField/')
from model.Initial import init_dnadiff,initial_dyna
from model.Solver import Solver
from model.Utils.Plot import drawExpDNA,plotDP,dna_exp_compare,dna_tdp_compare,plot_simulation
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation,FFMpegWriter
from functools import partial

para = {"m":128,"n":128,"dt":0.2,"max_iter":8000,
        "b1":0.14,"b2":0.4,"b3":-0.425,"b4":0.85,
         "Dd_0":0.3,"dd_0":7e-3,"ad_0":5e-4,
        "c1":0.6,"c2":0.6,"L1":1,"L2":1,"Critical":0.85,
        "alpha":0.008,"gamma":0,"beta":0,"base":0.3,"dna_level":0.5}
def initialize(data):
    eta,phi,chi = data
    dna = chi["0"]
    fig_main,axs_main = plt.subplots(2,3,figsize=(14,5.6),gridspec_kw={'width_ratios': [1,1,1],'height_ratios':[1,0.05]})
    axs_main[0, 0] = sns.heatmap(dna.real, ax=axs_main[0, 0], cmap="Reds", vmin=0, vmax=1, cbar=False)
    axs_main[0, 0].set_title("$\chi$: DNA concentration inside \n the DNA-Protein condensates")
    fig_main.colorbar(axs_main[0, 0].get_children()[0], cax=axs_main[1, 0], orientation="horizontal")
    axs_main[0, 1] = sns.heatmap(eta.real, ax=axs_main[0, 1], cmap="Greens", vmin=-1, vmax=1, cbar=False)
    axs_main[0, 1].set_title("$\eta$: Protein concentration inside \n the DNA-Protein condensates")
    fig_main.colorbar(axs_main[0, 1].get_children()[0], cax=axs_main[1, 1], orientation="horizontal")
    axs_main[0, 2] = sns.heatmap(phi.real, ax=axs_main[0, 2], cmap="viridis", vmin=-1, vmax=1, cbar=False)
    axs_main[0, 2].set_title("$\phi$: The hydrophobic and hydrophilic distributions \n inside the DNA-Protein condensates")
    fig_main.colorbar(axs_main[0, 2].get_children()[0], cax=axs_main[1, 2], orientation="horizontal")
    for ax in axs_main[0,:]:
        ax.tick_params(left=False, bottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig_main.suptitle(f"Simulation of FUS-ERG-DNA condensates")
    fig_main.tight_layout()
    return fig_main,axs_main
def animate_frame(i,data_sequence):
    eta = data_sequence[i][0]
    phi = data_sequence[i][1]
    dna = data_sequence[i][2]["0"]
    axs_main[0, 0].clear()
    axs_main[0, 1].clear()
    axs_main[0, 2].clear()
    axs_main[0, 0] = sns.heatmap(dna.real, ax=axs_main[0, 0], cmap="Reds", vmin=0, vmax=1, cbar=False)
    axs_main[0, 0].set_title("$\chi$: DNA concentration inside \n the DNA-Protein condensates")
    axs_main[0, 1] = sns.heatmap(eta.real, ax=axs_main[0, 1], cmap="Greens", vmin=-1, vmax=1, cbar=False)
    axs_main[0, 1].set_title("$\eta$: Protein concentration inside \n the DNA-Protein condensates")
    axs_main[0, 2] = sns.heatmap(phi.real, ax=axs_main[0, 2], cmap="viridis", vmin=-1, vmax=1, cbar=False)
    axs_main[0, 2].set_title(
        "$\phi$: The hydrophobic and hydrophilic distributions \n inside the DNA-Protein condensates")
    fig_main.suptitle(f"Simulation of FUS-ERG-DNA condensates under lower $b_1$ : time step {i*20}")
    fig_main.tight_layout()
    return fig_main
para_lb1 = para.copy()
para_lb1['b1'] = 0.04
eta,phi,chi,d_init,p_init = init_dnadiff(para_lb1,radius=13,dna_level_0=1)
init_order = {"eta":torch.Tensor(eta),"phi":torch.Tensor(phi),"chi":{"0":torch.Tensor(chi["0"])}}
solver = Solver(para_lb1,init_order,logger=None,target_region="eta-phi",DNA=True,crit=0.15)
solver.initialize()
record_seq = list(np.linspace(0,8000,200,dtype=int))
solver.solve(ifwrite=False,record_sequence=record_seq)
result = solver.getResult()
fig_main,axs_main = initialize(result[0])
results = list(map(result.get,record_seq))
ani = FuncAnimation(fig_main, partial(animate_frame,data_sequence=results), frames=200,interval=1)
ffwriter = FFMpegWriter(fps=10)
ani.save('animation_exp_lowerb1.mp4', writer=ffwriter)
