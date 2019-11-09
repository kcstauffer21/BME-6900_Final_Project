# Importing Packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.style as style
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go
import scipy


# Reading in the data
df_final_wide = pd.read_csv('./Data/102919_data_and_attributes.csv', encoding='utf8', engine='python')

# Styleizing Plots
style.use('seaborn-paper')
style.use('ggplot')
sns.set_style('whitegrid')

from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    name='test', 
    colors=['red','black','green']
)

# Creating smoker no smoker
df_final_wide.loc[(df_final_wide.tobacco_smoking_year_started == "[Not Available]") & (df_final_wide.tobacco_smoking_year_stopped == "[Not Available]"), "Smoker_yes_no"] = 'no'
df_final_wide.loc[~(df_final_wide.tobacco_smoking_year_started == "[Not Available]") & ~(df_final_wide.tobacco_smoking_year_stopped == "[Not Available]"), "Smoker_yes_no"] = 'yes'
df_final_wide.Smoker_yes_no.fillna('yes', inplace=True)

# Lung Location
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.slice(0,1) == 'R', "Lung_Location"] = "right"
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.slice(0,1) == 'L', "Lung_Location"] = "left"

# Lung Hemispheres
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.contains('Lower'), "Lung_Hemispheres"] = "Lower"
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.contains('Upper'), "Lung_Hemispheres"] = "Upper"

# Metastasis no metastasis column
df_final_wide.loc[(df_final_wide.loc[:, "ajcc_nodes_pathologic_pn"] == 'N0'), 'node_metastasis_yes_no'] = 'no'

df_final_wide.loc[~(df_final_wide.loc[:, "ajcc_nodes_pathologic_pn"] == 'N0'), 'node_metastasis_yes_no'] = 'yes'

# function inputs
data = df_final_wide
column_of_interest = "gender"

# Sorting the data by column of interest
data.sort_values(by=column_of_interest, ascending=False, inplace=True)
data.reset_index(inplace=True, drop=True)
df_svd = data.iloc[:,1: 12043].transpose().values
# df_svd = data.astype('float')
# running svd
u, s, v = np.linalg.svd(df_svd, full_matrices=False)
s = np.diag(s)
print(np.max(np.abs(df_svd-np.linalg.multi_dot([u,s,v]))))
unique, counts = np.unique(np.diag(s), return_counts = True)
counted_vals = dict(zip(unique, counts))
print(len(counted_vals))

# Plotting all v,s,s
fig, ax = plt.subplots(1, 3)
sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15, ax = ax[0])
# sns.heatmap(v[:, :], center=0, cmap=cmap, ax = ax[0])
# sns.heatmap(v[:, :], cmap=cmap, ax = ax[0])
sns.heatmap(u[:, :], cmap=cmap, vmin=-0.10, vmax=0.10, ax = ax[1])
sns.heatmap(s[:10, :10], cmap=cmap, center = 0, ax = ax[2])
plt.show()

# plotting v
sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15)
plt.show()
# plotting first 5 rows
sns.heatmap(v[0:5, :], cmap=cmap, center=0)

# pulling second row from v
v_df = pd.DataFrame(v[1, :])

v_df[column_of_interest] = df_final_wide.loc[:, column_of_interest].transpose()

sns.boxplot(x=v_df.loc[:, column_of_interest], y=v_df.iloc[:, 0])

# running stats
temp_stats=scipy.stats.mannwhitneyu(x = v_df[v_df.loc[:, column_of_interest] == 'yes'].iloc[:, 0], 
                         y = v_df[v_df.loc[:, column_of_interest] == 'no'].iloc[:, 0])
print(temp_stats)
