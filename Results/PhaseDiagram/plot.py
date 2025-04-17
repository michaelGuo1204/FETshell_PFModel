import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data_raw = pd.read_csv("result_dna_anno.csv").sort_values(by=['dna'],ascending=False)
#data_raw['dna'] = -data_raw['dna']
#data['ring_base'] = -data['ring_base']
data_raw = data_raw[data_raw['dna']!=0.44]
fig_2d,ax_2d = plt.subplots(figsize=(9,7))
markers = {"Y":'$\circ$',"N":'o',"E":'x'}
name = {"Y":"Shell","N":"Droplet","E":"No phase separation"}
for marker, d in data_raw.groupby('shell'):
    ax_2d.scatter(x=d['protein'], y=d['dna'],marker=markers[marker],s=100,label=name[marker])

ax_2d.set_xticks([0.2,0.64],['Less','More'])
ax_2d.set_yticks([0.55,0.97],['Less','More'])
ax_2d.set_xlabel("Protein")
ax_2d.set_ylabel("DNA")
fig_2d.legend(loc='right',bbox_to_anchor=(0.9,0.8))
fig_2d.savefig("phase_diagram2d.pdf",dpi=300)