import numpy as np
from skimage.draw import disk
from scipy.ndimage import gaussian_filter
import torch
def initial(para):
    n,m,base,crit = para["n"],para["m"],para["ring_base"],para["Critical"]
    rr,cc = disk((n/2,m/2),20)#13
    num_roi = len(rr)
    phi_hyphilic = np.zeros([n, m])
    phi_hyphilic[rr,cc] = base/5*4+ np.random.randn(num_roi)/100000
    phi_hyphobic = np.zeros([n, m])
    phi_hyphobic[rr,cc] = base/5*6+ np.random.randn(num_roi)/100000
    eta = phi_hyphobic + phi_hyphilic - crit
    phi = phi_hyphilic - phi_hyphobic
    rr_ring_outer,cc_ring_outer = disk((n/2,m/2),19)
    rr_ring_inner,cc_ring_inner = disk((n/2,m/2),17)
    mask = np.zeros([n,m])
    mask[rr_ring_outer,cc_ring_outer] = 1;mask[rr_ring_inner,cc_ring_inner] = 0
    rr,cc= np.where(mask==1)[0],np.where(mask==1)[1]
    phi[rr,cc]= 0.5
    drr,dcc = disk((n/2,m/2),15)
    eta = gaussian_filter(eta,sigma=3)
    phi = gaussian_filter(phi,sigma=3)
    chi = np.zeros([n, m])
    chi[drr,dcc] = 1
    chi = gaussian_filter(chi, sigma=3)
    print(phi.mean(),eta.mean(),(phi_hyphobic + phi_hyphilic).mean())
    eta = torch.Tensor(eta)
    phi = torch.Tensor(phi)
    chi = torch.Tensor(chi)
    return eta,phi,chi
def initial_dyna(para,radius=None):
    n,m,base,crit,dna_level = map(para.get,["n","m","base","Critical","dna_level"])
    if radius is None:
        radius = int(np.sqrt(n*m*base/(17*crit**2)))
    rr,cc = disk((n/2,m/2),radius)#13
    num_roi = len(rr)
    phi_hyphilic = np.zeros([n, m])
    phi_hyphilic[rr,cc] = base/6*5+ np.random.randn(num_roi)/100000
    phi_hyphobic = np.zeros([n, m])
    phi_hyphobic[rr,cc] = base/6*7+ np.random.randn(num_roi)/100000
    eta = phi_hyphobic + phi_hyphilic - crit
    phi = phi_hyphilic - phi_hyphobic
    rr_ring_outer,cc_ring_outer = disk((n/2,m/2),radius+1)
    rr_ring_inner,cc_ring_inner = disk((n/2,m/2),radius-1)
    mask = np.zeros([n,m])
    mask[rr_ring_outer,cc_ring_outer] = 1;mask[rr_ring_inner,cc_ring_inner] = 0
    rr,cc= np.where(mask==1)[0],np.where(mask==1)[1]
    phi_bar = phi.mean()
    head_value = (-phi_bar-1e-3)*(n*m)/(rr.shape[0])
    head_value = head_value if head_value >0.2 else 0.2
    phi[rr,cc]= head_value
    drr,dcc = disk((n/2,m/2),radius-3)
    eta = gaussian_filter(eta,sigma=3)
    phi = gaussian_filter(phi,sigma=3)
    chi = np.zeros([n, m])
    chi[drr,dcc] = dna_level
    chi = gaussian_filter(chi, sigma=3)
    print(phi.mean(),eta.mean(),(phi_hyphobic + phi_hyphilic).mean())
    eta = torch.Tensor(eta)
    phi = torch.Tensor(phi)
    chi = torch.Tensor(chi)
    return eta,phi,chi
def initial_tdp(para,radius=15,dna_level=0.5):
    n,m,base,crit = para["n"],para["m"],para["base"],para["Critical"]
    rr,cc = disk((n/2,m/2),radius)#13
    num_roi = len(rr)
    phi_hyphilic = np.zeros([n, m])
    phi_hyphilic[rr,cc] = base/6*5+ np.random.randn(num_roi)/100000
    phi_hyphobic = np.zeros([n, m]) 
    phi_hyphobic[rr,cc] = base/6*7+ np.random.randn(num_roi)/100000
    eta = phi_hyphobic + phi_hyphilic - crit
    phi = phi_hyphilic - phi_hyphobic
    eta = gaussian_filter(eta,sigma=3)
    phi = gaussian_filter(phi,sigma=3)
    dna = np.zeros([n,m])
    dna[rr,cc] = dna_level
    delta = gaussian_filter(dna,sigma=3)
    print(phi.mean(),eta.mean(),(phi_hyphobic + phi_hyphilic).mean())
    return eta,phi,delta,delta,delta

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
def init_dnadiff(para,radius=13,dna_level_0=0.5,dna_level_1=0.5,diffuse_steps=2000):
    n,m,base,crit = map(para.get,["n","m","base","Critical"])
    # generate disk/ring or mask
    disk_rr,disk_cc = disk((n/2,m/2),radius)#13
    disk_mask_ini = np.zeros([n,m])
    disk_mask_ini[disk_rr,disk_cc] = 1
    (rr_dna_ring_outer,cc_dna_ring_outer),(rr_dna_ring_inner,cc_dna_ring_inner) = disk((n/2,m/2),radius+5),disk((n/2,m/2),radius+2)
    dna_region = np.zeros_like(disk_mask_ini)
    dna_region[rr_dna_ring_outer,cc_dna_ring_outer] = 1
    mask = np.zeros([n,m])
    mask[rr_dna_ring_outer,cc_dna_ring_outer] = 1;mask[rr_dna_ring_inner,cc_dna_ring_inner] = 0
    dna_ring_rr,dna_ring_cc = np.where(mask==1)[0],np.where(mask==1)[1]
    # DNA diffusion
    coupled_filtered = gaussian_filter(dna_region,sigma=3)
    dna_0 = np.zeros_like(coupled_filtered)
    dna_0[dna_ring_rr,dna_ring_cc] = dna_level_0
    dna_initial = {"0":dna_0}
    if "Dd_1" in para.keys():
        dna_1 = np.zeros_like(disk_mask_ini)
        dna_1[rr_dna_ring_outer,cc_dna_ring_outer] = dna_level_1
        dna_initial.update({"1":dna_1})
    diffused_dna = dna_diffusion(dna_initial,coupled_filtered,para,0,diffuse_steps)
    # Prepare phi and eta
    invasive_dna = diffused_dna["0"] * disk_mask_ini
    if "Dd_1" in para.keys():
        invasive_dna += 0.2*(diffused_dna["1"] * disk_mask_ini)
        new_radius = np.sqrt((6*invasive_dna.sum()+np.pi*radius**2)/(np.pi))
        aux_dna_radius = np.sqrt((diffused_dna["1"].sum()/np.pi+(new_radius)**2))
        aux_ring_outer = disk((n/2,m/2),aux_dna_radius)
        aux_ring_inner = disk((n/2,m/2),new_radius)
        mask_aux = np.zeros([n,m])
        mask_aux[aux_ring_outer] = 1;mask_aux[aux_ring_inner] = 0
        aux_ring_rr,aux_ring_cc = np.where(mask_aux==1)[0],np.where(mask_aux==1)[1]
        aux_dna = np.zeros_like(disk_mask_ini)
        aux_dna[aux_ring_rr,aux_ring_cc] = 1
        new_dna_region = np.zeros_like(disk_mask_ini)
        new_dna_region[aux_ring_outer] = 1
        new_dna_region = gaussian_filter(new_dna_region,sigma=3)
        diffused_dna_1 = dna_diffusion({"1":aux_dna},new_dna_region,para,0,1500)
        diffused_dna["1"] = diffused_dna_1["1"]
    new_radius = np.sqrt((6*invasive_dna.sum()+np.pi*radius**2)/(np.pi))
    print(new_radius)
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
    return eta,phi,diffused_dna,dna_initial,disk_mask_ini

def init_phasediagram(para,radius=13,dna_level_0=0.5,base=0.5,diffuse_steps=2000):
    n,m= map(para.get,["n","m"])
    # generate disk/ring or mask
    disk_rr,disk_cc = disk((n/2,m/2),radius)#13
    disk_mask_ini = np.zeros([n,m])
    disk_mask_ini[disk_rr,disk_cc] = 1
    (rr_dna_ring_outer,cc_dna_ring_outer),(rr_dna_ring_inner,cc_dna_ring_inner) = disk((n/2,m/2),radius+5),disk((n/2,m/2),radius+2)
    dna_region = np.zeros_like(disk_mask_ini)
    dna_region[rr_dna_ring_outer,cc_dna_ring_outer] = 1
    mask = np.zeros([n,m])
    mask[rr_dna_ring_outer,cc_dna_ring_outer] = 1;mask[rr_dna_ring_inner,cc_dna_ring_inner] = 0
    dna_area = mask.sum()
    dna_ring_rr,dna_ring_cc = np.where(mask==1)[0],np.where(mask==1)[1]
    # DNA diffusion
    coupled_filtered = gaussian_filter(dna_region,sigma=3)
    dna_0 = np.zeros_like(coupled_filtered)
    dna_0[dna_ring_rr, dna_ring_cc] = dna_level_0 * 400 / dna_area
    dna_initial = {"0": dna_0}
    diffused_dna = dna_diffusion(dna_initial, coupled_filtered, para, 0, diffuse_steps)
    # Prepare phi and eta
    invasive_dna = diffused_dna["0"] * disk_mask_ini
    average_dna = (invasive_dna.sum() / (1.2 * disk_mask_ini.sum()))
    new_radius = round(np.sqrt((6*invasive_dna.sum()+np.pi*radius**2)/(np.pi)))
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
    effective_base = base*(2.2+2*average_dna-2.5*base)-0.1
    phi_hyphilic[disk_rr,disk_cc] = effective_base/6*5+ np.random.randn(num_roi)/100000
    phi_hyphobic[disk_rr,disk_cc] = effective_base/6*7+ np.random.randn(num_roi)/100000
    eta = phi_hyphobic + phi_hyphilic - (1.02 - average_dna)
    phi = phi_hyphilic - phi_hyphobic
    phi[ring_rr,ring_cc]=0.2
    eta = gaussian_filter(eta,sigma=3)
    phi = gaussian_filter(phi,sigma=3)
    print("DNA_level:{:.2f}, base:{:.2f},effective_base:{:.2f}, eta: {:.2f}, average_dna:{:.2f}, new radius: {:1f}".format(dna_level_0,base,effective_base,eta.mean(),average_dna,new_radius))
    return eta,phi,{"0":disk_mask_ini*dna_level_0}

def testPDInitialPara(para,radius=13,dna_level_0=0.5,dna_level_1=0.5,diffuse_steps=2000):
    n, m, base, crit = map(para.get, ["n", "m", "base", "Critical"])
    # generate disk/ring or mask
    disk_rr, disk_cc = disk((n / 2, m / 2), radius)  # 13
    disk_mask_ini = np.zeros([n, m])
    disk_mask_ini[disk_rr, disk_cc] = 1
    (rr_dna_ring_outer, cc_dna_ring_outer), (rr_dna_ring_inner, cc_dna_ring_inner) \
        = disk((n / 2, m / 2), radius + 5), disk((n / 2, m / 2), radius + 2)
    dna_region = np.zeros_like(disk_mask_ini)
    dna_region[rr_dna_ring_outer, cc_dna_ring_outer] = 1
    mask = np.zeros([n, m])
    mask[rr_dna_ring_outer, cc_dna_ring_outer] = 1
    mask[rr_dna_ring_inner, cc_dna_ring_inner] = 0
    dna_area = mask.sum()
    dna_ring_rr, dna_ring_cc = np.where(mask == 1)[0], np.where(mask == 1)[1]
    # DNA diffusion
    coupled_filtered = gaussian_filter(dna_region, sigma=3)
    dna_0 = np.zeros_like(coupled_filtered)
    dna_0[dna_ring_rr, dna_ring_cc] = dna_level_0 * 350 / dna_area
    dna_initial = {"0": dna_0}
    diffused_dna = dna_diffusion(dna_initial, coupled_filtered, para, 0, diffuse_steps)
    # Prepare phi and eta
    invasive_dna = diffused_dna["0"] * disk_mask_ini
    average_dna = (invasive_dna.sum() / (1.1 * disk_mask_ini.sum()))
    new_radius = np.sqrt((20 * invasive_dna.sum() + np.pi * (radius-2) ** 2) / (np.pi))*0.7
    return average_dna,new_radius,invasive_dna.sum()