import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy.fft import fft2,ifft2
from model.Solver import Solver
import seaborn as sns
from matplotlib import animation
class String_Solver:
    def __init__(self, start, end, intr, max_step, dt):
        self.interval = intr
        self.dt = dt
        self.method = 'RK4'
        self.start = start
        self.end = end
        self.max = max_step
        self.dim = start.shape
        self.stringInitiation(self.start, self.end, self.interval)

    def stringInitiation(self,start, end, interval):
        assert start.size == end.size
        dim = start.size
        ini_string = np.zeros((interval, dim))
        for i in range(dim):
            ini_string[:, i] = np.linspace(start[i], end[i], interval)
        self.x = ini_string

    def stringEvolution(self):
        if self.method == 'fEuler':
            for index, element in enumerate(self.x):
                self.x[index] = element - self.dt * self.graFunc(element)
        elif self.method == 'RK4':
            for index, element in enumerate(self.x):
                k1 = self.dt * self.graFunc(element)
                k2 = self.dt * self.graFunc(element + k1 / 2)
                k3 = self.dt * self.graFunc(element + k2 / 2)
                k4 = self.dt * self.graFunc(element + k3)
                self.x[index] = element - k1 / 6 - k2 / 3 - k3 / 3 - k4 / 6


    def stringReParam(self):
        s = np.zeros(self.x.shape[0])
        newstring = np.zeros(self.x.shape)
        for i in range(1, self.x.shape[0]):
            s[i] = (s[i - 1] + np.linalg.norm(self.x[i] - self.x[i - 1]))
        alpha = np.array(s / s[-1])
        origin_inter = np.linspace(0, 1, self.interval)
        for i in range(self.x.shape[1]):
            newstring[:, i] = np.interp(origin_inter, alpha, self.x[:, i])
        self.x = newstring

    def solve(self,graFunc):
        self.graFunc = graFunc
        for i in range(1000):
            self.stringEvolution()
            self.stringReParam()
        return self.x

class FFTStringSolver(String_Solver):
    def __init__(self, solver,**kwargs):
        super().__init__(**kwargs)
        assert isinstance(solver,Solver)
        assert solver._init
        self.solver = solver

    def stringInitiation(self,start, end, interval):
        self.x = np.zeros((self.interval,)+self.dim,dtype=complex)
        for i in range(self.dim[1]):
            for j in range(self.dim[2]):
                self.x[:, 0, i, j] = np.linspace(start[0, i, j], end[0, i, j], interval)
                self.x[:, 1, i, j] = np.linspace(start[1, i, j], end[1, i, j], interval)

    def stringEvolution(self):
        for index, element in enumerate(self.x):
            element = [torch.Tensor(element[0]),torch.Tensor(element[1])]
            self.x[index] = self.solver.fftGra(element)

    def stringReParam(self):
        s = np.zeros(self.x.shape[0])
        for i in range(1, self.x.shape[0]):
            s[i] = (s[i - 1] + np.linalg.norm(self.x[i] - self.x[i - 1]))
        alpha = np.array(s / s[-1])
        origin_inter = np.linspace(0, 1, self.interval)
        for i in range(self.dim[1]):
            for j in range(self.dim[2]):
                self.x[:, 0, i, j] = np.interp(origin_inter, alpha, self.x[:, 0, i, j])
                self.x[:, 1, i, j] = np.interp(origin_inter, alpha, self.x[:, 1, i, j])
    def solve(self,graFunc=None):
        for i in range(self.max):
            self.stringEvolution()
            self.stringReParam()
        return self.x
