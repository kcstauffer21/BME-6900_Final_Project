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
df_final_wide = df_final_wide.replace("[Not Available]", np.nan)


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


# -------------------------------------------------------------------------------------Running SVD and Plots
# function inputs
data = df_final_wide
columns_of_interest = df_final_wide.columns[12046:]
# #
# #
# # # v_row = 1
j = 0
i = 0


def plot_run_svd(data, column_of_interest, v_row):
    # Sorting the data by column of interest
    data.sort_values(by=column_of_interest, ascending=False, inplace=True)
    data.reset_index(inplace=True, drop=True)
    df_svd = data.iloc[:, 1: 12043].transpose().values

    # running svd
    u, s, v = np.linalg.svd(df_svd, full_matrices=False)
    s = np.diag(s)
    print(np.max(np.abs(df_svd - np.linalg.multi_dot([u, s, v]))))
    print(f"Dimensions of U: {u.shape}, Dimensions of S:{s.shape}, Dimensions of V:{v.shape}")
    unique, counts = np.unique(np.diag(s), return_counts=True)
    counted_vals = dict(zip(unique, counts))
    print(len(counted_vals))

    # Plotting all v,s,s
    fig, ax = plt.subplots(1, 3)
    ax0 = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15, ax=ax[0])
    ax1 = sns.heatmap(u[:, :], cmap=cmap, vmin=-0.10, vmax=0.10, ax=ax[1])
    ax2 = sns.heatmap(s[:, :], cmap=cmap, center=0, ax=ax[2])
    ax0.title.set_text('V Transposed Matrix')
    ax1.title.set_text('U Matrix')
    ax2.title.set_text('S Matrix')
    plt.show()
    # saving figure
    # plt.savefig(f"./Graphs/Heatmap_All_{column_of_interest}.png")

    # plotting v
    temp_v_heatmap = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15)
    plt.title("V Transposed Matrix")
    plt.show()
    # Saving figure
    temp_v_heatmap.figure.savefig(f"./Graphs/Heatmap_V_{column_of_interest}.png")

    # plotting first 5 rows
    temp_v_heatmap_first_five = sns.heatmap(v[0:5, :], cmap=cmap, center=0)
    plt.title("V Transposed Matrix First 5 rows")
    plt.show()
    # Saving figure
    temp_v_heatmap_first_five.figure.savefig(f"./Graphs/Heatmap_V_1to5_{column_of_interest}.png")

    # pulling row from v
    v_df = pd.DataFrame(v[v_row, :])
    v_df[column_of_interest] = df_final_wide.loc[:, column_of_interest]

    # Calculation the counts and medians for column_of_interest
    medians = v_df.groupby(column_of_interest).median().values
    column_of_interest_counts = v_df[column_of_interest].value_counts().values
    column_of_interest_counts = [str(x) for x in column_of_interest_counts.tolist()]
    column_of_interest_counts = ["n= " + i for i in column_of_interest_counts]

    # running stats Mann Whitney
    if len(data.loc[:, column_of_interest].unique()) == 2:
        temp_stats = scipy.stats.mannwhitneyu(
            x=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[0]].iloc[:, 0],
            y=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[1]].iloc[:, 0])

    # Making boxplot
    if temp_stats[1] < 0.05:
        temp_bp = sns.boxplot(x=v_df.loc[:, column_of_interest], y=v_df.iloc[:, 0])
        pos = range(len(column_of_interest_counts))
        for tick, label in zip(pos, temp_bp.get_xticklabels()):
            temp_bp.text(pos[tick], medians[tick] + 0.01, column_of_interest_counts[tick],
                         ha='center', fontsize=12, color='w', weight='semibold')
        temp_bp.set(ylabel=f"V of {column_of_interest}, Row: {v_row}", title=f"Boxplot of {column_of_interest}")
        temp_bp.text(x=-0.3, y=-0.3, s=f"p={temp_stats[1]}", fontsize=10, ha='center', va='bottom')
        plt.show()

        # Saving figure
        temp_bp.figure.savefig(f"./Graphs/bp_{column_of_interest}_Row_{v_row}.png")
        print(temp_stats)


def plot_run_svd_all(data, columns_of_interest):
    multiple_unique = list()
    start_time = datetime.datetime.now()
    for i in range(len(columns_of_interest)):
        # print(i)
        column_of_interest = columns_of_interest[i]
        all_stats = dict()
        data = data[(data.loc[:, column_of_interest].notnull())]
        for j in range(1, len(data.loc[:, column_of_interest])):
            # print(j)
            # running stats Mann Whitney
            if len(data.loc[:, column_of_interest].unique()) == 2:
                # Sorting the data by column of interest
                data.sort_values(by=column_of_interest, ascending=False, inplace=True)
                data.reset_index(inplace=True, drop=True)
                df_svd = data.iloc[:, 1: 12043].transpose().values

                # running svd
                u, s, v = np.linalg.svd(df_svd, full_matrices=False)
                s = np.diag(s)
                # print(np.max(np.abs(df_svd - np.linalg.multi_dot([u, s, v]))))
                # print(f"Dimensions of U: {u.shape}, Dimensions of S:{s.shape}, Dimensions of V:{v.shape}")
                # unique, counts = np.unique(np.diag(s), return_counts=True)
                # counted_vals = dict(zip(unique, counts))
                # print(len(counted_vals))

                # pulling row from v
                v_df = pd.DataFrame(v[j, :])
                v_df[column_of_interest] = data.loc[:, column_of_interest]
                temp_stats = scipy.stats.mannwhitneyu(
                    x=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[0]].iloc[:, 0],
                    y=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[1]].iloc[:, 0])
                all_stats.update({f"{v_df.columns[1]}_{j}": temp_stats[1]})
            else:
                break
        try:
            min_pval = min(all_stats.keys(), key=lambda k: all_stats[k])
            if all_stats[min_pval] < 0.05:
                # Calculation the counts and medians for column_of_interest
                medians = v_df.groupby(column_of_interest).median().values
                column_of_interest_counts = v_df[column_of_interest].value_counts().values
                column_of_interest_counts = [str(x) for x in column_of_interest_counts.tolist()]
                column_of_interest_counts = ["n= " + i for i in column_of_interest_counts]

                # Making boxplot
                temp_bp = sns.boxplot(x=v_df.loc[:, column_of_interest], y=v_df.iloc[:, 0])
                pos = range(len(column_of_interest_counts))
                for tick, label in zip(pos, temp_bp.get_xticklabels()):
                    temp_bp.text(pos[tick], medians[tick] + 0.01, column_of_interest_counts[tick],
                                 ha='center', fontsize=12, color='w', weight='semibold')
                temp_bp.set(ylabel=f"V of {min_pval}", title=f"Boxplot of {column_of_interest}")
                temp_bp.text(x=-0.3, y=v_df.iloc[:, 0].min() + -0.3, s=f"p={all_stats[min_pval]}", fontsize=10,
                             ha='center', va='bottom')
                plt.show()

                # Plotting all v,u,s
                # fig, ax = plt.subplots(1, 3)
                # ax0 = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15, ax=ax[0])
                # ax1 = sns.heatmap(u[:, :], cmap=cmap, vmin=-0.10, vmax=0.10, ax=ax[1])
                # ax2 = sns.heatmap(s[:, :], cmap=cmap, center=0, ax=ax[2])
                # ax0.title.set_text('V Transposed Matrix')
                # ax1.title.set_text('U Matrix')
                # ax2.title.set_text('S Matrix')
                # plt.show()

                # saving figure
                # plt.savefig(f"./Graphs/Heatmap_All_{column_of_interest}.png")

                # plotting v
                temp_v_heatmap = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15)
                plt.title(f"V Transposed Matrix sorted by {column_of_interest}")
                plt.show()
                # Saving figure
                temp_v_heatmap.figure.savefig(f"./Graphs/Heatmap_V_{column_of_interest}.png")

                # plotting first 5 rows
                # temp_v_heatmap_first_five = sns.heatmap(v[0:5, :], cmap=cmap, center=0)
                # plt.title(f"V Transposed Matrix sorted by {column_of_interest} First 5 rows")
                # plt.show()
                # # Saving figure
                # temp_v_heatmap_first_five.figure.savefig(f"./Graphs/Heatmap_V_1to5_{column_of_interest}.png")

                # Saving figure
                temp_bp.figure.savefig(f"./Graphs/bp_{min(all_stats)}_.png")
                # print(temp_stats)
        except ValueError:
            multiple_unique.append(column_of_interest)
            print(f"Error on {column_of_interest}")
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    return multiple_unique


def plot_run_svd_all_test(data, columns_of_interest):
    multiple_unique = list()
    start_time = datetime.datetime.now()
    for i in range(len(columns_of_interest)):
        # print(i)
        column_of_interest = columns_of_interest[i]
        all_stats = dict()
        # data = data[(data.loc[:, column_of_interest].notnull())]
        for j in range(len(data.loc[:, column_of_interest])):
            # print(j)
            # running stats Mann Whitney

            # Sorting the data by column of interest
            data.sort_values(by=column_of_interest, ascending=False, inplace=True)
            data.reset_index(inplace=True, drop=True)
            df_svd = data.iloc[:, 1: 12043].transpose().values

            # running svd
            u, s, v = np.linalg.svd(df_svd, full_matrices=False)
            s = np.diag(s)
            # print(np.max(np.abs(df_svd - np.linalg.multi_dot([u, s, v]))))
            # print(f"Dimensions of U: {u.shape}, Dimensions of S:{s.shape}, Dimensions of V:{v.shape}")
            # unique, counts = np.unique(np.diag(s), return_counts=True)
            # counted_vals = dict(zip(unique, counts))
            # print(len(counted_vals))

            # pulling row from v
            v_df = pd.DataFrame(v[j, :])
            v_df[column_of_interest] = data.loc[:, column_of_interest]
        try:
            temp_stats = scipy.stats.mannwhitneyu(
                x=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[0]].iloc[:, 0],
                y=v_df[v_df.loc[:, column_of_interest] == v_df.loc[:, column_of_interest].unique()[1]].iloc[:, 0])
            all_stats.update({f"{v_df.columns[1]}_{j}": temp_stats[1]})

            min_pval = min(all_stats.keys(), key=lambda k: all_stats[k])
            if all_stats[min_pval] < 0.05:
                # Calculation the counts and medians for column_of_interest
                medians = v_df.groupby(column_of_interest).median().values
                column_of_interest_counts = v_df[column_of_interest].value_counts().values
                column_of_interest_counts = [str(x) for x in column_of_interest_counts.tolist()]
                column_of_interest_counts = ["n= " + i for i in column_of_interest_counts]

                # Making boxplot
                temp_bp = sns.boxplot(x=v_df.loc[:, column_of_interest], y=v_df.iloc[:, 0])
                pos = range(len(column_of_interest_counts))
                for tick, label in zip(pos, temp_bp.get_xticklabels()):
                    temp_bp.text(pos[tick], medians[tick] + 0.01, column_of_interest_counts[tick],
                                 ha='center', fontsize=12, color='w', weight='semibold')
                temp_bp.set(ylabel=f"V of {min_pval}", title=f"Boxplot of {column_of_interest}")
                temp_bp.text(x=-0.3, y=v_df.iloc[:, 0].min() + -0.3, s=f"p={all_stats[min_pval]}", fontsize=10,
                             ha='center', va='bottom')
                plt.show()

                # Plotting all v,u,s
                # fig, ax = plt.subplots(1, 3)
                # ax0 = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15, ax=ax[0])
                # ax1 = sns.heatmap(u[:, :], cmap=cmap, vmin=-0.10, vmax=0.10, ax=ax[1])
                # ax2 = sns.heatmap(s[:, :], cmap=cmap, center=0, ax=ax[2])
                # ax0.title.set_text('V Transposed Matrix')
                # ax1.title.set_text('U Matrix')
                # ax2.title.set_text('S Matrix')
                # plt.show()

                # saving figure
                # plt.savefig(f"./Graphs/Heatmap_All_{column_of_interest}.png")

                # plotting v
                temp_v_heatmap = sns.heatmap(v[:, :], cmap=cmap, vmin=-0.15, vmax=0.15)
                plt.title(f"V Transposed Matrix sorted by {column_of_interest}")
                plt.show()
                # Saving figure
                temp_v_heatmap.figure.savefig(f"./Graphs/Heatmap_V_{column_of_interest}.png")

                # plotting first 5 rows
                # temp_v_heatmap_first_five = sns.heatmap(v[0:5, :], cmap=cmap, center=0)
                # plt.title(f"V Transposed Matrix sorted by {column_of_interest} First 5 rows")
                # plt.show()
                # # Saving figure
                # temp_v_heatmap_first_five.figure.savefig(f"./Graphs/Heatmap_V_1to5_{column_of_interest}.png")

                # Saving figure
                temp_bp.figure.savefig(f"./Graphs/bp_{min(all_stats)}_.png")
                # print(temp_stats)
        except:
            multiple_unique.append(column_of_interest)
            print(f"Error on {column_of_interest}")
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    return multiple_unique




# plot_run_svd(df_final_wide, 'node_metastasis_yes_no', 1)

temp = plot_run_svd_all(df_final_wide, df_final_wide.columns[12046:])
