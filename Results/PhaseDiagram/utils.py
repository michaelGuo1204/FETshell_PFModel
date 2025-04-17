import numpy as np
from skimage.draw import disk
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
def dna_diffusion(dna_initial,region,para,crit,steps):
    region[region<crit] = 0
    region_hat = np.fft.fft2(region)
    M,N,step = para["m"],para["n"],0.1
    xx,yy = np.meshgrid(2 * np.pi / M * np.concatenate([np.arange(0, M / 2), np.arange(-M / 2, 0)])
                         , 2 * np.pi / N * np.concatenate([np.arange(0, N / 2), np.arange(-N / 2, 0)]))
    nk2 = xx ** 2 + yy ** 2
    region_x = np.fft.ifft2(xx*region_hat).real/(region+1e-3)
    region_y = np.fft.ifft2(yy*region_hat).real/(region+1e-3)
    chi_hat = {key:np.fft.fft2(value) for key,value in dna_initial.items()}
    for i in range(steps):
        last_chihat = chi_hat
        for key,value in dna_initial.items():
            Dd= para["Dd_{}".format(key)]
            chi_x = np.fft.ifft2(xx * last_chihat[key]).real
            chi_y = np.fft.ifft2(yy * last_chihat[key]).real
            chi_hat[key] = (last_chihat[key] + Dd*step*np.fft.fft2(region_x*chi_x+region_y*chi_y))/(1+Dd*step*nk2)
    chi = {key:np.fft.ifft2(value).real for key,value in chi_hat.items()}
    return chi
def init_dnadiff(para,radius,dna_level=1):
    n,m,base,crit = map(para.get,["n","m","base","Critical"])
    # generate disk/ring or mask
    disk_rr,disk_cc = disk((n/2,m/2),radius)#13
    disk_mask_ini = np.zeros([n,m])
    disk_mask_ini[disk_rr,disk_cc] = 1
    (rr_dna_ring_outer,cc_dna_ring_outer),(rr_dna_ring_inner,cc_dna_ring_inner) = disk((n/2,m/2),radius+5),disk((n/2,m/2),radius)
    dna_region = np.zeros_like(disk_mask_ini)
    dna_region[rr_dna_ring_outer,cc_dna_ring_outer] = 1
    mask = np.zeros([n,m])
    mask[rr_dna_ring_outer,cc_dna_ring_outer] = 1;mask[rr_dna_ring_inner,cc_dna_ring_inner] = 0
    dna_ring_rr,dna_ring_cc = np.where(mask==1)[0],np.where(mask==1)[1]
    # DNA diffusion
    coupled_filtered = gaussian_filter(dna_region,sigma=3)
    dna = np.zeros_like(coupled_filtered)
    dna[dna_ring_rr,dna_ring_cc] = dna_level
    dna_initial = {"0":dna,"1":dna.copy()} if "Dd_1" in para.keys() else {"0":dna}
    diffused_dna = dna_diffusion(dna_initial,coupled_filtered,para,0,1500)
    #possible_region[possible_region<0.2] = 0
    # Prepare phi and eta
    invasive_dna = diffused_dna["0"] * disk_mask_ini
    new_radius = np.sqrt((10*invasive_dna.sum()+np.pi*radius**2)/(np.pi))
    if "Dd_1" in para.keys():
        aux_dna_radius = np.sqrt((diffused_dna["1"].sum()/np.pi+(new_radius+2)**2))
        aux_ring_outer = disk((n/2,m/2),aux_dna_radius)
        aux_ring_inner = disk((n/2,m/2),new_radius+2)
        mask_aux = np.zeros([n,m])
        mask_aux[aux_ring_outer] = 1;mask_aux[aux_ring_inner] = 0
        aux_ring_rr,aux_ring_cc = np.where(mask_aux==1)[0],np.where(mask_aux==1)[1]
        aux_dna = np.zeros_like(disk_mask_ini)
        aux_dna[aux_ring_rr,aux_ring_cc] = 1
        diffused_dna["1"] = aux_dna

    print("D:{},New R:{}".format(para["Dd_0"],new_radius))
    disk_rr,disk_cc = disk((n/2,m/2),new_radius)#13
    disk_mask = np.zeros([n,m])
    disk_mask[disk_rr,disk_cc] = 1
    rr_ring_outer,cc_ring_outer = disk((n/2,m/2),new_radius+2)
    rr_ring_inner,cc_ring_inner = disk((n/2,m/2),new_radius-2)
    mask = np.zeros([n,m])
    mask[rr_ring_outer,cc_ring_outer] = 1;mask[rr_ring_inner,cc_ring_inner] = 0
    ring_rr,ring_cc= np.where(mask==1)[0],np.where(mask==1)[1]
    phi_hyphilic = np.zeros([n,m])
    phi_hyphobic = np.zeros([n,m])
    num_roi = len(disk_rr)
    phi_hyphilic[disk_rr,disk_cc] = base/6*5+ np.random.randn(num_roi)/100000
    phi_hyphobic[disk_rr,disk_cc] = base/6*7+ np.random.randn(num_roi)/100000
    eta = phi_hyphobic + phi_hyphilic - crit
    phi = phi_hyphilic - phi_hyphobic
    phi[ring_rr,ring_cc]=0.2
    eta = gaussian_filter(eta,sigma=3)
    phi = gaussian_filter(phi,sigma=3)
    print(phi.mean(),eta.mean(),(phi_hyphobic + phi_hyphilic).mean())
    return eta,phi,diffused_dna,dna_initial,disk_mask_ini
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