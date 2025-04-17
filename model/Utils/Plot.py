import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter
import multiprocessing as mp
import tqdm
from math import floor
from model.Solver import Solver

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])

def dna_tdp_compare(ax,d_dna,start=67,end=86,multiplier=39):
   file =  "../stack/RGG.csv"
   il = 21
   return dna_Compare(ax,d_dna,file,start,end,il,multiplier)

def dna_exp_compare(ax,d_dna,start=72,end=92,multiplier=285):
    file =  "../stack/FUSERG.csv"
    il = 11
    return dna_Compare(ax,d_dna,file,start,end,il,multiplier)

def dna_Compare(ax,d_dna,file,start,end,il,multiplier):
    exp_data = pd.read_csv(file,index_col=0)
    data_list = []
    for i in exp_data.columns:
        for j in exp_data.index:
            data_list.append(pd.Series([i,j,exp_data.loc[j,i]]))
    mod_df = pd.concat(data_list,axis=1).T
    mod_df.columns = ["location","case","value"]
    ax = sns.lineplot(data=mod_df,x="location",y="value",label="Exp",ax=ax)
    simu_data = d_dna[64,start:end]
    interval = end-start
    inter_data = np.interp(np.arange(0,interval,interval/il),np.arange(0,interval,1),simu_data)*multiplier
    pd.DataFrame(inter_data).to_csv('./dna.csv')
    ax = sns.lineplot(inter_data,c='red',label="Simu",ax=ax,color='black')
    # Turns off grid on the left Axis.
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.set_ylabel("Intensity")
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    return ax
def plot_protein(eta_out,phi_out):
    fig,axs = plt.subplots(2,2,figsize=(10,6),gridspec_kw={'width_ratios': [1, 1],'height_ratios':[1,0.05]})
    axs[0,0]=sns.heatmap(eta_out.real,ax=axs[0,0],cmap="Greens",vmin=-1,vmax=1,cbar=False)
    axs[0,0].set_title("$\eta$")
    fig.colorbar(axs[0,0].get_children()[0],cax=axs[1,0],orientation="horizontal")
    axs[0,1]=sns.heatmap(phi_out.real,ax=axs[0,1],cmap="viridis",vmin=-1,vmax=1,cbar=False)
    axs[0,1].set_title("$\phi$")
    fig.colorbar(axs[0,1].get_children()[0],cax=axs[1,1],orientation="horizontal")
    for ax in axs[0]:
        ax.tick_params(left=False, bottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    return fig
def plot_dna(chi,datatype='exp',figsize=(10,5.5)):
    fig_dna,axs_dna = plt.subplots(2,2,figsize=figsize,gridspec_kw={'width_ratios': [1, 1],'height_ratios':[1,0.01]})
    axs_dna[0,0]=sns.heatmap(chi["0"].real,ax=axs_dna[0,0],cmap="Reds",vmin=0,vmax=1,cbar=False)
    axs_dna[0,0].set_title("$\chi$")
    axs_dna[0,0].get_xaxis().set_visible(False)
    axs_dna[0,0].get_yaxis().set_visible(False)
    axs_dna[0,0].tick_params(left=False, bottom=False)
    fig_dna.colorbar(axs_dna[0,0].get_children()[0],cax=axs_dna[1,0],orientation="horizontal")
    gs = axs_dna[0,1].get_gridspec()
    for ax in axs_dna[:,1]:
        ax.remove()
    axbig = fig_dna.add_subplot(gs[:, 1])
    if datatype == 'exp':
        axbig = dna_exp_compare(axbig,chi["0"].real)
    else:
        axbig = dna_tdp_compare(axbig,chi["0"].real)
    fig_dna.tight_layout()
    return fig_dna

def plot_simulation(result,dna_index="0",figsize=(10,5.6),dna_alpha=0.7):
    eta,phi,chi = result
    dna = chi[dna_index]
    dna_cmap = "Reds" if dna_index == "0" else "Blues"
    fig_main,axs_main = plt.subplots(2,2,figsize=figsize,gridspec_kw={'width_ratios': [1, 1],'height_ratios':[1,0.05]})
    axs_main[0, 0] = sns.heatmap(eta.real, ax=axs_main[0, 0], cmap="Greens", vmin=-1, vmax=1, cbar=False)
    #axs_main[0, 0].set_title("Protein $\eta$")
    fig_main.colorbar(axs_main[0, 0].get_children()[0], cax=axs_main[1, 0], orientation="horizontal")
    axs_main[0, 1] = sns.heatmap(chi[dna_index].real, ax=axs_main[0, 1], cmap=dna_cmap, vmin=0, vmax=1, cbar=False)
    #axs_main[0, 1].set_title("DNA $\chi$")
    fig_main.colorbar(axs_main[0, 1].get_children()[0], cax=axs_main[1, 1], orientation="horizontal")
    fig_aux,axs_aux = plt.subplots(2,2,figsize=figsize,gridspec_kw={'width_ratios': [1, 1],'height_ratios':[1,0.05]})
    axs_aux[0, 0] = sns.heatmap(phi.real, ax=axs_aux[0, 0], cmap="viridis", vmin=-1, vmax=1, cbar=False)
    #axs_aux[0, 0].set_title("Protein separation $\phi$")
    axs_aux[0,1] = sns.heatmap(chi[dna_index].real,ax=axs_aux[0,1],cmap=dna_cmap,vmin=0,vmax=1,cbar=False,alpha=dna_alpha)
    axs_aux[0,1] = sns.heatmap(eta.real,ax=axs_aux[0,1],cmap="Greens",vmin=-1,vmax=1,cbar=False,alpha=0.3)
    fig_aux.colorbar(axs_aux[0, 0].get_children()[0], cax=axs_aux[1, 0], orientation="horizontal")
    axs_aux[1,1].set_visible(False)
    for ax in [axs_main[0,0],axs_main[0,1],axs_aux[0,0],axs_aux[0,1]]:
        ax.tick_params(left=False, bottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig_aux.tight_layout()
    fig_main.tight_layout()
    return fig_main,fig_aux
def plot_simu_womerge(result,dna_index="0",figsize=(14,5.6)):
    eta, phi, chi = result
    dna = chi[dna_index]
    fig_main, axs_main = plt.subplots(2, 3, figsize=figsize,
                                      gridspec_kw={'width_ratios': [1, 1, 1], 'height_ratios': [1, 0.05]})
    axs_main[0, 0] = sns.heatmap(dna.real, ax=axs_main[0, 0], cmap="Reds", vmin=0, vmax=1, cbar=False)
    fig_main.colorbar(axs_main[0, 0].get_children()[0], cax=axs_main[1, 0], orientation="horizontal")
    axs_main[0, 1] = sns.heatmap(eta.real, ax=axs_main[0, 1], cmap="Greens", vmin=-1, vmax=1, cbar=False)
    fig_main.colorbar(axs_main[0, 1].get_children()[0], cax=axs_main[1, 1], orientation="horizontal")
    axs_main[0, 2] = sns.heatmap(phi.real, ax=axs_main[0, 2], cmap="viridis", vmin=-1, vmax=1, cbar=False)
    fig_main.colorbar(axs_main[0, 2].get_children()[0], cax=axs_main[1, 2], orientation="horizontal")
    for ax in axs_main[0, :]:
        ax.tick_params(left=False, bottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig_main.tight_layout()
    return fig_main

def drawExpDNA(d_dna,coulped_compound,cmap='Greens'):
    y = np.arange(0,70)
    x = np.arange(0,26)
    xx,yy = np.meshgrid(y,x)
    zz = d_dna[64:90,30:100]
    cc = coulped_compound[64:90,30:100]
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(xx,yy,zz,alpha=0.2,shade=True,color='#FF6F91',edgecolor='#FF6F91',lw=0.5,rstride=1, cstride=1,)
    ax.bar(y,zz[0,:],zdir='y',alpha=0.6,color='#9C5865')
    #ax.plot(y,zz[0,:],zdir='y',color='#4D8076')
    ax.contourf(xx,yy,cc,zdir='z',offset=-0.05,alpha=0.7,cmap=cmap)
    #ax.set_zlim(-0.05,0.5)
    ax.set_xlim(0,70)
    ax.set_ylim(0,26)
    ax.set_box_aspect([1,1,1])
    set_axes_equal(ax)
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=30., azim=-60, roll=0)
    return fig

def sampleworker(para):
    img,phi,ri = para
    u = np.linspace(np.pi / 2, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 200)
    sinv = np.sin(v)
    cosv = np.cos(v)
    cor_x = np.around(ri * cosv).astype(int) + 64
    cor_y = np.around(ri * sinv).astype(int) + 64
    target = img[cor_x, cor_y]
    roi = np.where(target>=img.quantile(0.25))
    color = phi[cor_x, cor_y]
    x = ri * np.outer(np.cos(u), sinv[roi])
    y = ri * np.outer(np.sin(u), sinv[roi])
    z = ri * np.outer(np.ones(np.size(u)), cosv[roi])
    c_protein = np.outer(np.ones(np.size(u)), color[roi])
    return np.array([x.flatten(),y.flatten(),z.flatten(),c_protein.flatten()])

def plotsphere(eta,phi,start,end=30):
    fig = plt.figure(figsize=(10,10))
    eta = gaussian_filter(eta, sigma=1)
    phi = gaussian_filter(phi, sigma=1)
    ax = fig.add_subplot(projection='3d')
    r = np.linspace(start,end,3*(end-start))
    working_list = [(eta.copy(),phi.copy(),ri) for ri in r]
    result_list = []
    bar = tqdm.tqdm(total=len(working_list))
    with mp.Pool(processes=12) as pool:
        for result in pool.map(sampleworker, working_list):
            result_list.append(result)
            bar.update(1)
    resultarray= np.concatenate(result_list,axis=1)
    xx = resultarray[0]
    yy = resultarray[1]
    zz = resultarray[2]
    cc = resultarray[3]
    # Plot the surface
    #ax.scatter(xx.flatten(), yy.flatten(), zz.flatten(),alpha=0.05,c=cc_dna.flatten(),cmap='Reds')
    ax.scatter(xx, yy, zz,alpha=0.05,c=cc,cmap='viridis',vmin=-1,vmax=1)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim(-end,end)
    ax.set_ylim(-end,end)
    ax.set_zlim(-end,end)
    ax.set_aspect('equal')
    ax.view_init(elev=10., azim=45, roll=0)
    return fig
def plotDP(dna_initial,protein,dna_index="0"):
    dna_initial = {key:gaussian_filter(value,sigma=3) for key,value in dna_initial.items()}
    protein = gaussian_filter(protein,sigma=3)
    fig_main,axs_main = plt.subplots(2,2,figsize=(10,6),gridspec_kw={'width_ratios': [1, 1],'height_ratios':[1,0.05]})
    axs_main[0, 0] = sns.heatmap(protein, ax=axs_main[0, 0], cmap="Greens", vmin=0, vmax=1, cbar=False)
    axs_main[0, 0].set_title("Protein")
    fig_main.colorbar(axs_main[0, 0].get_children()[0], cax=axs_main[1, 0], orientation="horizontal")
    axs_main[0, 1] = sns.heatmap(dna_initial[dna_index], ax=axs_main[0, 1], cmap="Reds", vmin=0, vmax=1, cbar=False)
    axs_main[0, 1].set_title("DNA")
    fig_main.colorbar(axs_main[0, 1].get_children()[0], cax=axs_main[1, 1], orientation="horizontal")
    for ax in axs_main[0]:
        ax.tick_params(left=False, bottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig_main.tight_layout()
    return fig_main
