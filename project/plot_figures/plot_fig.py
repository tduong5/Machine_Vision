import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv('plot_figures/obj_detect.csv')
sns.set_style("darkgrid")
sns_plot = sns.lineplot(hue="region", style="event", data=df)
plt.title('Object Accuracy')
plt.xlabel('Time (seconds)')
plt.ylabel('Accuracy (Percentage)')
plt.show()
sns_plot.figure.savefig("plot_figures/output.png")
