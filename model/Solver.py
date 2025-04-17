import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.loggers import MLFlowLogger
from torch import fft as fft
from scipy.ndimage.filters import minimum_filter, maximum_filter
class Solver:
    def __init__(self,param,initial,logger:MLFlowLogger|None,target_region="eta",crit=0,DNA=False):
        self._param = param
        self._init = False
        self.eta = None
        self.phi = None
        self.eta_hat = None
        self.phi_hat = None
        if DNA:
            self.hasDNA=True
            self.chi = {}
            self.chi_hat = {}
            self.target_region = target_region
        else:
            self.hasDNA = False
        self.initial = initial
        self.logger = logger
        self.multiple_dna = False
        self.crit = crit
        self.info = {"error":[],"energy":[]}
        self.results = {}


    def initialize(self):
        self.multiple_dna = False
        M = self._param["m"]
        N = self._param["n"]
        C = 32
        self.xx, self.yy = torch.meshgrid(2 * torch.pi / M * torch.cat([torch.arange(0, M / 2), torch.arange(-M / 2, 0)])
                         , 2 * torch.pi / N * torch.cat([torch.arange(0, N / 2), torch.arange(-N / 2, 0)]))
        self.nk2 = self.xx**2 + self.yy**2
        xf, yf = torch.meshgrid(
            2 * torch.pi / (M * C) * torch.cat([torch.arange(0, M * C / 2), torch.arange(-(M * C) / 2, 0)])
            , 2 * torch.pi / (N * C) * torch.cat([torch.arange(0, N * C / 2), torch.arange(-(N * C) / 2, 0)]))
        self.nk2f = xf * xf + yf * yf
        self.nk2f[0,0] = 1
        self.generateStaticVar()
        self._init = True
        if self.logger is not None:
            self.logger.log_hyperparams(self._param)
        if callable(self.initial):
            if self.hasDNA:
                self.eta, self.phi,self.chi  = self.initial(self._param)
                num_dna = len(self.chi.keys())
                try:
                    _ = map(self._param.get, ["Dd_{}".format(num_dna), "dd_{}".format(num_dna), "ad_{}".format(num_dna)])
                except KeyError:
                    raise KeyError("Please specify the coefficient for all the DNA")
                self.chi_hat = {key: fft.fft2(value) for key, value in self.chi.items()}
            else:
                self.eta, self.phi = self.initial(self._param)
        elif isinstance(self.initial,dict):
            if self.hasDNA:
                self.eta, self.phi, self.chi = self.initial["eta"],self.initial["phi"],self.initial["chi"]
                self.chi_hat = {key: fft.fft2(value) for key, value in self.chi.items()}
            else:
                self.eta, self.phi = self.initial["eta"],self.initial["phi"]
        self.eta_hat, self.phi_hat = fft.fft2(self.eta), fft.fft2(self.phi)
        #self.plotStates("Initial",write=True)
        #self.displayPotential(self._param)

    def generateStaticVar(self):
        c1, c2, M, N = map(self._param.get,["c1", "c2", "m", "n"])
        b1, b2, b3, b4 = map(self._param.get,["b1", "b2", "b3", "b4"])
        alpha, beta, gamma = map(self._param.get,["alpha", "beta", "gamma"])
        L1,L2,step = map(self._param.get,["L1","L2","dt"])
        A11 = 1 + gamma * step * L1 + self.nk2 * (step * c1 * L1 * self.nk2 - step * L1)
        A12 = beta * step * L1 + step * b1 * L1 * self.nk2
        A21 = beta * step * L2 + step * b1 * L2 * self.nk2
        A22 = 1 + alpha * step * L2 + self.nk2 * (step * c2 * L2 * self.nk2 - step * L2)
        Delta = A11 * A22 - A12 * A21
        B11 = A22 / Delta
        B12 = -step * L1 * self.nk2 * B11
        B13 = -A12 / Delta
        B14 = -step * L2 * self.nk2 * B13
        B21 = -A21 / Delta
        B22 = -step * L1 * self.nk2 * B21
        B23 = A11 / Delta
        B24 = -step * L2 * self.nk2 * B23
        B11[0, 0] = 1;self.B11 = B11
        B12[0, 0] = 0;self.B12 = B12
        B13[0, 0] = 0;self.B13 = B13
        B14[0, 0] = 0;self.B14 = B14
        B21[0, 0] = 0;self.B21 = B21
        B22[0, 0] = 0;self.B22 = B22
        B23[0, 0] = 1;self.B23 = B23
        B24[0, 0] = 0;self.B24 = B24

    def checkEnergy(self):
        assert self.eta is not None and self.phi is not None
        eta = self.eta
        phi = self.phi
        eta_bar = torch.mean(eta)
        phi_bar = torch.mean(phi)
        c1, c2, M, N = map(self._param.get, ["c1", "c2", "m", "n"])
        b1, b2, b3, b4 = map(self._param.get, ["b1", "b2", "b3", "b4"])
        alpha, beta, gamma = map(self._param.get, ["alpha", "beta", "gamma"])
        C = 32
        Fs_dif = (c1 / 2 / M / N * torch.sum(self.nk2 * abs(fft.fft2(eta)) ** 2)
                  + c2 / 2 / M / N * torch.sum(self.nk2 * abs(fft.fft2(phi)) ** 2))
        Fs_local = (torch.sum((eta ** 2 - 1) ** 2) / 4 + torch.sum((phi ** 2 - 1) ** 2) / 4
                    + b1 * torch.sum(eta * phi) - b2 / 2 * torch.sum((eta * phi ** 2))
                    - b3 / 2 * torch.sum(phi * eta ** 2)
                    + b4 / 2 * torch.sum(eta ** 2 * phi ** 2))
        eta_hat = torch.fft.fft2(eta - eta_bar, s=[C * M, C * N])
        phi_hat = torch.fft.fft2(phi - phi_bar, s=[C * M, C * N])
        Fl = (alpha / 2 * torch.sum(abs(phi_hat) ** 2 / self.nk2f)
              + beta * torch.sum(eta_hat * torch.conj(phi_hat) / self.nk2f)
              + gamma / 2 * torch.sum(abs(eta_hat) ** 2 / self.nk2f)) / C / C / M / N
        F = Fs_dif + Fs_local + Fl
        return F.real

    def displayPotential(self):
        b1, b2, b3, b4 = map(self._param.get, ["b1", "b2", "b3", "b4"])
        w = lambda eta, phi: (eta ** 2 - 1) ** 2 / 4 + (
                    phi ** 2 - 1) ** 2 / 4 + b1 * eta * phi - b2 * eta * phi ** 2 / 2 - b3 * phi * eta ** 2 / 2 + b4 * eta ** 2 * phi ** 2 / 2
        x = np.linspace(-1.5, 1.5, 1000)
        y = np.linspace(-1.5, 1.5, 1000)
        xx, yy = np.meshgrid(x, y)
        ww = w(xx, yy)
        mn = minimum_filter(ww, size=3, mode='wrap')
        result = np.where(mn == ww)
        true_cor = np.zeros([len(result[0]), len(result)])
        fig,ax = plt.subplots(figsize=(6,6))
        ax.contourf(x, y, ww, 200, alpha=0.5)
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("$\phi$")
        for i in range(result[0].size):
            ax.scatter(x[result[1][i]], y[result[0][i]], marker='X', s=100, c='w')
            true_cor[i, :] = np.array([x[result[1][i]], y[result[0][i]]])
        return fig
    def fftGra(self,element):
        eta = element[0];phi = element[1]
        b1, b2, b3, b4 = map(self._param.get, ["b1", "b2", "b3", "b4"])
        f1hat = fft.fft2(
            eta ** 3 - b2 / 2 * phi ** 2 - b3 * eta * phi + b4 * eta * phi ** 2)
        f2hat = fft.fft2(
            phi ** 3 - b2 * eta *phi - b3 / 2 * eta ** 2 + b4 * eta ** 2 * phi)
        eta_hat = fft.fft2(eta)
        phi_hat = fft.fft2(phi)
        eta_hat = self.B11 * eta_hat+ self.B12 * f1hat + self.B13 * phi_hat + self.B14 * f2hat
        phi_hat = self.B21 * eta_hat+ self.B22 * f1hat + self.B23 * phi_hat + self.B24 * f2hat
        eta_new = fft.ifft2(eta_hat)
        phi_new = fft.ifft2(phi_hat)
        return np.array([eta_new,phi_new])
    def fftsolve(self):
        b1, b2, b3, b4,step = map(self._param.get, ["b1", "b2", "b3", "b4","dt"])
        f1hat = fft.fft2(self.eta ** 3 - b2 / 2 * self.phi ** 2 - b3 * self.eta * self.phi + b4 * self.eta * self.phi ** 2)
        f2hat = fft.fft2(self.phi ** 3 - b2 * self.eta * self.phi - b3 / 2 * self.eta ** 2 + b4 * self.eta ** 2 * self.phi)
        last_etahat = self.eta_hat
        last_phihat = self.phi_hat
        self.eta_hat = self.B11 * last_etahat + self.B12 * f1hat + self.B13 * last_phihat + self.B14 * f2hat
        self.phi_hat = self.B21 * last_etahat + self.B22 * f1hat + self.B23 * last_phihat + self.B24 * f2hat
        last_eta = self.eta
        last_phi = self.phi
        self.eta = fft.ifft2(self.eta_hat)
        self.phi = fft.ifft2(self.phi_hat)
        if self.hasDNA:
            last_chihat = self.chi_hat
            if self.target_region == "eta":
                region = self.eta.real.clone()
            elif self.target_region == "-phi":
                region = -self.phi.real.clone()
            elif self.target_region == "phi":
                region = self.phi.real.clone()
            elif self.target_region == "eta-phi":
                region = self.eta.real.clone() - self.phi.real.clone()
            elif self.target_region == "eta+phi":
                region = self.phi.real.clone() +1.5* self.eta.real.clone()
            region[region<self.crit] = 0
            region_hat = fft.fft2(region)
            region_x = fft.ifft2(self.xx*region_hat).real/(region+1e-3)
            region_y = fft.ifft2(self.yy*region_hat).real/(region+1e-3)
            for key in self.chi_hat.keys():
                Dd,dd,ad = map(self._param.get,["Dd_{}".format(key),"dd_{}".format(key),"ad_{}".format(key)])
                chi_x = fft.ifft2(self.xx * last_chihat[key]).real
                chi_y = fft.ifft2(self.yy * last_chihat[key]).real
                self.chi_hat[key] = (last_chihat[key] + Dd*step*fft.fft2(region_x*chi_x+region_y*chi_y)+ad*region_hat)/(1+Dd*step*self.nk2+dd*step)
                self.chi[key] = fft.ifft2(self.chi_hat[key])
        error_s = torch.norm(last_eta - self.eta, 'fro') + torch.norm(last_phi - self.phi, 'fro')
        return error_s
    def solve(self,ifwrite=False,record_sequence=None):
        for i in range(self._param['max_iter']+1):
            if record_sequence is not None and i in record_sequence:
                self.results.update({i: (self.eta.clone(), self.phi.clone(), {key:value.clone() for key,value in self.chi.items()})})
            error = self.fftsolve()
            if i% 1000 ==0:
                self.logEnergy(i,error,ifwrite)
        if ifwrite:
            return self.plotStates("Final energy{}".format(self.info["energy"][-1]),write=True)
        else:
            return None
        
    def logEnergy(self,step,error,ifwrite=False):
        energy = self.checkEnergy()
        if self.logger is not None:
            self.logger.log_metrics({"Energy":energy,"Error":error},step = step)
        #print("eta mean:{},phi mean{}".format(torch.mean(self.eta),torch.mean(self.phi)))
        self.info["energy"].append(energy.item())
        self.info["error"].append(error)
    def plotStates(self,comments="",write=False):
        size = self._param["m"]/self._param["n"]
        n_fig = 3 if self.hasDNA else 2
        fig, axs = plt.subplots(nrows=1, ncols=n_fig, figsize=(5*n_fig*size,5))
        cbar_ax = fig.add_axes([.95,.3,.02,.4])
        plt.suptitle(comments)
        eta = self.eta.real
        phi = self.phi.real
        data_list={"\eta (Protein)":eta,"\phi (LLPS)":phi}
        if self.hasDNA:
            chi = [a.real for a in self.chi.values()]
            data_list.update({"\chi (DNA)":chi[0]})
        for ax,(key,value) in zip(axs,data_list.items()):
            make_cbar = (key == '\phi (LLPS)')
            ax = sns.heatmap(pd.DataFrame(value.T), cmap='viridis', vmin=-1, vmax=1, cbar=make_cbar,ax=ax,cbar_ax=None if not make_cbar else cbar_ax)
            ax.set_title("${}$".format(key))
            ax.tick_params(left=False, bottom=False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if write:
            return fig
        else: pass

    def updateParam(self,param):
        self._param = param
    def getResult(self):
        if self.results == {}:
            return {"Final":(self.eta,self.phi,self.chi)}
        else:
            return self.results





