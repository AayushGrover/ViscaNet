import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
        'Exp': ['Exp.1', 'Exp.2', 'Exp.1', 'Exp.2', 'Exp.1', 'Exp.2', 'Exp.1', 'Exp.2', 'Exp.1', 'Exp.2'],
        'Metric': ['Precision', 'Precision', 'Recall', 'Recall', 'AUC', 'AUC', 'F1', 'F1', 'MCC', 'MCC'],
        'Score' : [0.9846, 0.9856, 0.989275, 0.989847, 0.99011, 0.984137, 0.98695, 0.9877585, 0.979288, 0.96931275]
}
df = pd.DataFrame(data, columns=['Exp', 'Metric', 'Score'])

plt.figure(figsize=(10, 8))
splot = sns.barplot(x="Metric", y="Score", hue="Exp", data=df)
plt.ylabel("Scores", size=14)
plt.xlabel("Metrics", size=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(prop={'size': 12})
plt.title("Results", size=18)
plt.ylim(0.95, 1.0)
for p in splot.patches:
    splot.annotate(format(p.get_height(), '.3f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   size=15,
                   xytext = (0, -12), 
                   textcoords = 'offset points')

plt.savefig('./diagrams/results.png')
