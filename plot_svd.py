# Importing Packages
import datetime

import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Stylizing Plots
style.use('seaborn-paper')
style.use('ggplot')
sns.set_style('whitegrid')

# Creating color map
cmap = LinearSegmentedColormap.from_list(
    name='test',
    colors=['red', 'black', 'green']
)

# Reading in the data
df_final_wide = pd.read_csv('./Data/102919_data_and_attributes.csv', encoding='utf8', engine='python')
# (133, 12151)
df_final_wide.shape

# Cleaning data by removing all columns that contain all [Not Available]
df_final_wide = df_final_wide.loc[:, ~(df_final_wide == "[Not Available]").all()]

# (133, 12100) drops 51 columns
df_final_wide.shape

# ------------------------------------------------------------------------------Creating more attribute
# Creating smoker no smoker
df_final_wide.loc[(df_final_wide.tobacco_smoking_year_started == "[Not Available]") & (
        df_final_wide.tobacco_smoking_year_stopped == "[Not Available]"), "Smoker_yes_no"] = 'no'
df_final_wide.loc[~(df_final_wide.tobacco_smoking_year_started == "[Not Available]") & ~(
        df_final_wide.tobacco_smoking_year_stopped == "[Not Available]"), "Smoker_yes_no"] = 'yes'
df_final_wide.Smoker_yes_no.fillna('yes', inplace=True)

# Lung Location
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.slice(0, 1) == 'R', "Lung_Location"] = "right"
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.slice(0, 1) == 'L', "Lung_Location"] = "left"

# Lung Hemispheres
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.contains('Lower'), "Lung_Hemispheres"] = "Lower"
df_final_wide.loc[df_final_wide.anatomic_organ_subdivision.str.contains('Upper'), "Lung_Hemispheres"] = "Upper"

# Metastasis no metastasis column
df_final_wide.loc[(df_final_wide.loc[:, "ajcc_nodes_pathologic_pn"] == 'N0'), 'node_metastasis_yes_no'] = 'no'
df_final_wide.loc[~(df_final_wide.loc[:, "ajcc_nodes_pathologic_pn"] == 'N0'), 'node_metastasis_yes_no'] = 'yes'

# Replacing all [Not Available] with na's
df_final_wide = df_final_wide.replace("[Not Available]", np.nan)

# -------------------------------------------------------------------------------------Running SVD and Plots
data = df_final_wide
columns_of_interest = df_final_wide.columns[12046:]


def plot_run_svd_all(data, columns_of_interest):
    """
    :param data: Pass in the dataframe of interest
    :param columns_of_interest: The attribute only columns, no gene expression data
    :return: A list of of columns with more than one unique values.
    """
    multiple_unique = list()
    start_time = datetime.datetime.now()
    for i in range(len(columns_of_interest)):
        i = 1
        # print(f"i={i}")
        column_of_interest = columns_of_interest[i]
        all_stats = dict()
        if (len(data.loc[:, column_of_interest].unique()) == 2 and data.loc[:,
                                                                   column_of_interest].isna().sum() == 0) or (
                len(data.loc[:, column_of_interest].unique()) == 3 and data.loc[:,
                                                                       column_of_interest].isna().sum() > 0):
            # Sorting the data by column of interest
            data.sort_values(by=column_of_interest, ascending=False, inplace=True)
            data.reset_index(inplace=True, drop=True)
            df_svd = data.iloc[:, 1: 12043].transpose().values

            # running svd
            u, s, v = np.linalg.svd(df_svd, full_matrices=False)

            # Running statistics
            for j in range(len(v[1:, 0])):
                v_df = pd.DataFrame(v[j, :])
                v_df[column_of_interest] = data.loc[:, column_of_interest]
                # running stats
                temp_stats = scipy.stats.mannwhitneyu(
                    x=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[0]].iloc[:, 0],
                    y=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[1]].iloc[:, 0])
                all_stats.update({f"{v_df.columns[1]}_{j}": temp_stats[1]})

            min_pval = min(all_stats, key=all_stats.get)
            if all_stats[min_pval] < 0.05:
                row_number = min_pval.split('_')[-1]
                # Calculation the counts and medians for column_of_interest
                column_of_interest_counts = v_df[column_of_interest].value_counts().values
                column_of_interest_counts = [str(x) for x in column_of_interest_counts.tolist()]
                column_of_interest_counts = ["n= " + i for i in column_of_interest_counts]

                # Making boxplot
                temp_bp = sns.boxplot(x=v_df.loc[:, column_of_interest], y=v_df.iloc[:, 0])
                pos = range(len(column_of_interest_counts))
                for tick, label in zip(pos, temp_bp.get_xticklabels()):
                    temp_bp.text(pos[tick], v_df.iloc[:, 0].min() + -0.5, column_of_interest_counts[tick],
                                 ha='center', fontsize=12, color='black', weight='semibold')
                temp_bp.set(ylabel=f"V of {min_pval}", title=f"Boxplot of {column_of_interest}")
                temp_bp.text(x=-0.3, y=v_df.iloc[:, 0].min() + -0.3, s=f"p={all_stats[min_pval]}", fontsize=10,
                             ha='center', va='bottom')
                plt.show()

                # plotting v
                temp_v_heatmap = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15)
                plt.title(f"V Transposed Matrix sorted by {column_of_interest}")
                plt.show()
                # Saving figure
                temp_v_heatmap.figure.savefig(f"./Graphs/Heatmap_V_{column_of_interest}.png")

                # Saving figure
                temp_bp.figure.savefig(f"./Graphs/bp_{min_pval}_.png")

                # Doing U magic
                u_df = pd.DataFrame(u)
                u_df["Genes"] = data.transpose().index[1:12043]
                u_df.sort_values(by=int(row_number), ascending=False, inplace=True)
                u_df.reset_index(inplace=True, drop=True)
                sns.barplot(x=u_df[int(row_number)], y=u_df.index)
                # .plot(u_df[int(row_number)], u_df.index)
                plt.title(f"Gene Expression of U sorted by column: {row_number}")
                plt.savefig(f'./Graphs/U_sortedby_{row_number}_gene_expression.png', bbox_inches='tight')
                plt.show()
                sorted_genes = u_df.Genes
                sorted_genes.to_csv(f"./Sorted U genes/U_sorted_Row_{row_number}_{column_of_interest}.csv", index=False)

                temp_u_heatmap = sns.heatmap(u[:, :], cmap=cmap, center=0)
                plt.title(f"U Matrix Sorted by Row {row_number}")
                plt.show()
                # Saving figure
                temp_u_heatmap.figure.savefig(f"./Graphs/Heatmap_U_{column_of_interest}_Row_{row_number}.png")

            else:
                multiple_unique.append(column_of_interest)
                print(f"Error on {column_of_interest}")

    # Plotting sigma matrix
    plt.plot(s[1:])
    plt.title("Eigenvalue Decomposition (Sigma Matrix)")
    plt.savefig('./Graphs/s_matrix.png', bbox_inches='tight')
    plt.show()

    s_temp = pd.DataFrame(s)
    temp_bar = sns.barplot(x=s_temp.index[1:], y=s_temp.iloc[1:, 0], data=s_temp, color='red')
    plt.title("Eigenvalue Decomposition (Sigma Matrix)")
    temp_bar.set(xlabel="Position", ylabel="Eigenvalue Decomposition")
    plt.savefig('./Graphs/s_matrix_bar.png', bbox_inches='tight')
    plt.show()

    s_temp = np.diag(s)
    sns.heatmap(s_temp[:, :], vmin=-0.65, vmax=0.65, cmap=cmap)
    plt.title("Eigenvalue Decomposition (Sigma Matrix)")
    plt.savefig('./Graphs/s_matrix_heatmap.png', bbox_inches='tight')
    plt.show()


    end_time = datetime.datetime.now()
    print(end_time - start_time)
    return multiple_unique


temp = plot_run_svd_all(df_final_wide, df_final_wide.columns[12046:])
