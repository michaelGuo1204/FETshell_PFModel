from model.Solver import Solver
from model.Utils.Plot import plot_simulation
from model.Initial import init_phasediagram,testPDInitialPara
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import multiprocessing
import pandas as pd
para = {"m":128,"n":128,"dt":0.2,"max_iter":4000,
        "b1":0.14,"b2":0.4,"b3":-0.425,"b4":0.85,
        "Dd_0":2e-1,"dd_0":7e-3,"ad_0":8e-4,
        "c1":0.6,"c2":0.6,"L1":1,"L2":1,"Critical":1,
        "alpha":0.008,"gamma":0,"beta":0,"base":0.3}
def worker(args):
    base, dna_level = args
    eta,phi,diffused_dna = init_phasediagram(para,base=base,dna_level_0=dna_level)
    init_order = {"phi":torch.Tensor(phi),"eta":torch.Tensor(eta),"chi":{"0":torch.Tensor(diffused_dna["0"])}}
    solver = Solver(para,init_order,logger=None,target_region="-phi",DNA=True,crit=0.5)
    solver.initialize()
    solver.solve(ifwrite=False,record_sequence=[0,4000])
    result = solver.getResult()
    final = result[4000]
    fig_final1,fig_final2 = plot_simulation(final)
    fig_final1.savefig("./fig/{:.2f}-{:.2f}.png".format(base,dna_level))
    return [base, dna_level]

result_list = []
radius_list = [0.18 + 0.06 * i for i in range(9)]
dna_level_list = [0.44 + 0.08*i for i in range(8)]
working_list = [(i,j) for i in radius_list for j in dna_level_list]
with multiprocessing.Pool(processes=8) as pool:
    for result in pool.map(worker,working_list):
        result_list.append(result)
result_df = pd.DataFrame(result_list,columns=["ring_base","critical_base"])
result_df.to_csv("./result_dna.csv")
#result_df = pd.DataFrame(result_list,columns=["radius","dna_level","average_dna","new_radius","invasive_dna"])
#fig,ax = plt.subplots(1,3,figsize=(18,6))
#sns.heatmap(result_df.pivot(index="dna_level",columns="radius",values="average_dna").sort_index(ascending=False),ax=ax[0])
#sns.heatmap(result_df.pivot(index="dna_level",columns="radius",values="new_radius").sort_index(ascending=False),ax=ax[1])
#sns.heatmap(result_df.pivot(index="dna_level",columns="radius",values="invasive_dna").sort_index(ascending=False),ax=ax[2])
#plt.show()
