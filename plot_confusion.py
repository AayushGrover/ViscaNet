import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

array1 = [771, 17, 12, 1170]
array2 = [35081, 320, 222, 20736]

names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

names1 = np.asarray([(name+'\n'+str(val)) for name, val in zip(names, array1)]).reshape(2,2)
names2 = np.asarray([(name+'\n'+str(val)) for name, val in zip(names, array2)]).reshape(2,2)

array1 = np.asarray(array1).reshape(2,2)
array2 = np.asarray(array2).reshape(2,2)

fig, ax =plt.subplots(1,2, figsize=(10,4))
sns.heatmap(array1, ax=ax[1], cmap='YlGnBu', annot=names1, fmt="", annot_kws={'size':11}, linewidths=.5)
ax[1].set_title('Exp. 2')
sns.heatmap(array2, ax=ax[0], cmap='YlGnBu', annot=names2, fmt="", annot_kws={'size':11}, linewidths=.5)
ax[0].set_title('Exp. 1')

ax[0].set_xlabel('Prediction', fontsize=12)
ax[1].set_xlabel('Prediction', fontsize=12)
ax[0].set_ylabel('Target', fontsize=12)

plt.tight_layout()
fig.savefig('../diagrams/confusion.png')