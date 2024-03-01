# -- coding: utf-8 --
# @Time : 2024/2/25 18:50
# @Author : Stephanie
# @Email : sunc696@gmail.com
# @File : human_vis.py

'''
## AI knowledge level ##
import numpy as np
import matplotlib.pyplot as plt


font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 20,
         'weight':'bold',
         }
font2 = {'family': 'Calibri',
         'weight': 'normal',
         'size': 15,
         'weight':'bold',
         }

data = np.genfromtxt('../XAI_human_study.csv', delimiter=',', dtype=str, encoding='utf-8')

data = [item for item in data if item.isdigit()]

counts = {x: data.count(x) for x in set(data)}

labels = ['Weak Knowledge', 'Moderate Knowledge', 'Strong Knowledge']
values = [counts.get(str(i), 0) for i in range(1, 4)]
print(values)

plt.bar(labels, values, color=['green', 'orange', 'blue'])

plt.title('Distribution of AI Knowledge Levels')
plt.xlabel('Knowledge Level')
plt.ylabel('Number of Responses')

plt.show()

'''

'''
## Radar chart

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

font1 = {'family': 'Arial',
         # 'weight': 'normal',
         'size': 15,
         'weight': 'bold',
         }
#
# df = pd.read_csv('../XAI_human_study.csv')
#
# rows_with_3 = df[df.iloc[:, 0] == 3]
# rows_with_2 = df[df.iloc[:, 0] == 2]
# rows_with_1 = df[df.iloc[:, 0] == 1]
#
# # Compute the mean for the columns 13 to 18 in these selected rows
# means3 = rows_with_3.iloc[:, 1:6].mean()
# means2 = rows_with_2.iloc[:, 1:6].mean()
# means1 = rows_with_1.iloc[:, 1:6].mean()
#
# # Print the mean scores for each method
# print(means3, means2, means1)
#
# exit(0)

# key feature
data = {
    'Advanced': [3, 5.57, 4.07, 6.36, 5.36, 6.43],
    'Intermediate': [2.09, 5.43, 4.29, 5.77, 4.94, 5.71],
    'Novice': [1.67, 6, 4.33, 5.78, 5.56, 6.67]
}

# # trust
# data = {
#     'Advanced': [-1.93, 0.29, -1.14, 1.21, 0.57, 1.57],
#     'Intermediate': [-2.8, 0.83, -0.54, 0.77, 0.23, 1.2],
#     'Novice': [-2.22, 1.56, 0.11, 0.67, 1.56, 2.56]
# }

categories = ['IG', 'LRP', 'Grad-CAM', 'SHAP', 'S-XAI','AS-XAI']
N = len(categories)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

data['Novice'] += data['Novice'][:1]
data['Intermediate'] += data['Intermediate'][:1]
data['Advanced'] += data['Advanced'][:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))


ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)


for label, values in data.items():
    ax.fill(angles, values, label=label, alpha=0.18)
    ax.plot(angles, values)


ax.set_thetagrids(np.degrees(angles[:-1]), categories)
for label, angle in zip(ax.get_xticklabels(), angles):
    if angle in (0, np.pi):
        label.set_horizontalalignment('center')
    elif 0 < angle < np.pi:
        label.set_horizontalalignment('left')
    else:
        label.set_horizontalalignment('right')


ax.yaxis.grid(True, linestyle='dashed')

ax.spines['polar'].set_visible(True)
ax.spines['polar'].set_linewidth(2)

ax.set_title(f'Human Evaluation of XAI-Presented Feature Accuracy',font1, pad=14)
# ax.set_title(f'Evaluation of Human Trust in XAI',font1, pad=13)

ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontname='Arial', fontsize=13)

ax.legend(loc='upper right', bbox_to_anchor=(1.11, 1.05), prop ='Arial',fontsize=13)

# plt.savefig(f'../lime_save_2kinds/trust_ASSESSMENT.tiff',bbox_inches='tight', dpi=300)
plt.savefig(f'../lime_save_2kinds/key feature_ASSESSMENT.tiff',bbox_inches='tight', dpi=300)

plt.show()
'''

# '''
## Stacked Bar Graph (global vs local)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../local.csv')

fig_columns = ['Fig1.1', 'Fig2.1', 'Fig3.1', 'Fig4.1', 'Fig5.1', 'Fig6.1']
prop = {}

for col in fig_columns:
    zeros_proportion = (df[col] == 0).mean()
    ones_proportion = (df[col] == 1).mean()
    prop[col] = {'0': zeros_proportion, '1': ones_proportion}

print(prop)

proportions = {}

# Loop over each 'Fig' column
for fig in ['Fig1.1', 'Fig2.1', 'Fig3.1', 'Fig4.1', 'Fig5.1', 'Fig6.1']:
    # Initialize sub-dictionary for each 'Fig'
    proportions[fig] = {'0': {}, '1': {}}
    # Total number of zeros and ones in the current 'Fig' column
    total_zeros = df[fig].value_counts()[0]
    total_ones = df[fig].value_counts()[1]
    # Loop over each unique value in 'AI' column
    for val in df['AI'].unique():
        # Count of zeros for the current 'AI' value in the current 'Fig'
        count_0 = df[(df['AI'] == val) & (df[fig] == 0)].shape[0]
        # Proportion of zeros for the current 'AI' value in the current 'Fig'
        proportions[fig]['0'][val] = count_0 / total_zeros * prop[fig]['0']
        # Count of ones for the current 'AI' value in the current 'Fig'
        count_1 = df[(df['AI'] == val) & (df[fig] == 1)].shape[0]
        # Proportion of ones for the current 'AI' value in the current 'Fig'
        proportions[fig]['1'][val] = count_1 / total_ones * prop[fig]['1']

print(proportions['Fig6.1']['1'])
font={'family': 'Arial',
         'weight': 'normal',
         'size': 15,
         # 'weight': 'bold',
         }

font3={'family': 'Arial',
         # 'weight': 'normal',
         'size': 18,
         'weight': 'bold',
         }

data_adjusted = np.array([
    # Local - Making these values negative
    [-proportions['Fig1.1']['0'][1], -proportions['Fig2.1']['0'][1], -proportions['Fig3.1']['0'][1], -proportions['Fig4.1']['0'][1], -proportions['Fig5.1']['0'][1], -proportions['Fig6.1']['0'][1]],  # Weak AI understanding - Local
    [-proportions['Fig1.1']['0'][2], -proportions['Fig2.1']['0'][2], -proportions['Fig3.1']['0'][2], -proportions['Fig4.1']['0'][2], -proportions['Fig5.1']['0'][2], -proportions['Fig6.1']['0'][2]],  # Moderate AI understanding - Local
    [-proportions['Fig1.1']['0'][3], -proportions['Fig2.1']['0'][3], -proportions['Fig3.1']['0'][3], -proportions['Fig4.1']['0'][3], -proportions['Fig5.1']['0'][3], -proportions['Fig6.1']['0'][3]],  # Strong AI understanding - Local
    # Global - Keeping these values positive
    [proportions['Fig1.1']['1'][1], proportions['Fig2.1']['1'][1], proportions['Fig3.1']['1'][1], proportions['Fig4.1']['1'][1], proportions['Fig5.1']['1'][1], proportions['Fig6.1']['1'][1]],  # Weak AI understanding - Global
    [proportions['Fig1.1']['1'][2], proportions['Fig2.1']['1'][2], proportions['Fig3.1']['1'][2], proportions['Fig4.1']['1'][2], proportions['Fig5.1']['1'][2], proportions['Fig6.1']['1'][2]],  # Moderate AI understanding - Global
    [proportions['Fig1.1']['1'][3], proportions['Fig2.1']['1'][3], proportions['Fig3.1']['1'][3], proportions['Fig4.1']['1'][3], proportions['Fig5.1']['1'][3], proportions['Fig6.1']['1'][3]],  # Strong AI understanding - Global
])

data_adjusted_t = data_adjusted.T

fig, ax = plt.subplots(figsize=(13, 8.5))

n_groups = data_adjusted_t.shape[0]
width = 0.4

ind = np.arange(n_groups)

p1 = ax.barh(ind, data_adjusted_t[:, 0], width, label='Acquainted - Local', color = '#61B4CF')
p2 = ax.barh(ind, data_adjusted_t[:, 1], width, left=data_adjusted_t[:, 0], label='Skilled - Local', color = '#3394B6')
p3 = ax.barh(ind, data_adjusted_t[:, 2], width, left=data_adjusted_t[:, 0] + data_adjusted_t[:, 1], label='Proficient - Local', color = '#027399')
p4 = ax.barh(ind + width, data_adjusted_t[:, 3], width, label='Acquainted - Global', color = '#8FC1B5')
p5 = ax.barh(ind + width, data_adjusted_t[:, 4], width, left=data_adjusted_t[:, 3], label='Skilled - Global', color = '#589A8D')
p6 = ax.barh(ind + width, data_adjusted_t[:, 5], width, left=data_adjusted_t[:, 3] + data_adjusted_t[:, 4], label='Proficient - Global', color = '#007566')

ax.axvline(0, color='grey', linewidth=1.5)
ax.set_xlabel('Percentage', fontname='Arial', fontsize=19, labelpad=17)
ax.set_title('Human Understanding of XAI Methods (Global/Local)', font3, pad=13)
ax.set_yticks(ind + width / 2, fontname='Arial', fontsize=18)
ax.set_yticklabels(['IG', 'LRP', 'Grad-CAM', 'SHAP', 'S-XAI', 'AS-XAI'], fontname='Arial', fontsize=18)

ax.legend(prop=font)

ax.text(-0.8, -1.02, 'Local', ha='right', va='center',fontname='Arial', fontsize=19, color='#016B8F', fontweight='bold')
ax.text(0.72, -1.02, 'Global', ha='left', va='center',fontname='Arial', fontsize=19, color='#007061', fontweight='bold')

ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels(['0.8', '0.6', '0.4', '0.2', '0', '0.2', '0.4', '0.6', '0.8'], fontname='Arial', fontsize=18)

plt.savefig(f'../lime_save_2kinds/global_vs_local.tiff',bbox_inches='tight', dpi=300)

plt.show()
# '''

'''
## Stacked Bar Graph(Common Semantic Space Purity)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('../local.csv')

fig_columns = ['Fig5.2', 'Fig6.2']
prop = {}

for col in fig_columns:
    zeros_proportion = (df[col] == 0).mean()
    ones_proportion = (df[col] == 1).mean()
    twos_proportion = (df[col] == 2).mean()
    threes_proportion = (df[col] == 3).mean()
    prop[col] = {'0': zeros_proportion, '1': ones_proportion, '2': twos_proportion, '3': threes_proportion}

print(prop)

proportions = {}

for fig in ['Fig5.2', 'Fig6.2']:
    # Initialize sub-dictionary for each 'Fig'
    proportions[fig] = {'0': {}, '1': {}, '2': {}, '3': {}}
    # Total number of zeros and ones in the current 'Fig' column
    total_zeros = df[fig].value_counts()[0]
    total_ones = df[fig].value_counts()[1]
    total_twos = df[fig].value_counts()[2]
    total_threes = df[fig].value_counts()[3]
    # Loop over each unique value in 'AI' column
    for val in df['AI'].unique():
        # Count of zeros for the current 'AI' value in the current 'Fig'
        count_0 = df[(df['AI'] == val) & (df[fig] == 0)].shape[0]
        # Proportion of zeros for the current 'AI' value in the current 'Fig'
        proportions[fig]['0'][val] = count_0 / total_zeros * prop[fig]['0']
        # Count of ones for the current 'AI' value in the current 'Fig'
        count_1 = df[(df['AI'] == val) & (df[fig] == 1)].shape[0]
        # Proportion of ones for the current 'AI' value in the current 'Fig'
        proportions[fig]['1'][val] = count_1 / total_ones *prop[fig]['1']
        # Count of twos for the current 'AI' value in the current 'Fig'
        count_2 = df[(df['AI'] == val) & (df[fig] == 2)].shape[0]
        # Proportion of threes for the current 'AI' value in the current 'Fig'
        proportions[fig]['2'][val] = count_2 / total_twos * prop[fig]['2']
        # Count of threes for the current 'AI' value in the current 'Fig'
        count_3 = df[(df['AI'] == val) & (df[fig] == 3)].shape[0]
        # Proportion of threes for the current 'AI' value in the current 'Fig'
        proportions[fig]['3'][val] = count_3 / total_threes * prop[fig]['3']

font={'family': 'Arial',
         'weight': 'normal',
         'size': 8,
         # 'weight': 'bold',
         }

font3={'family': 'Arial',
         # 'weight': 'normal',
         'size': 9,
         'weight': 'bold',
         }

data_adjusted = np.array([
    # Local - Making these values negative
    [-proportions['Fig5.2']['0'][1], -proportions['Fig5.2']['1'][1], -proportions['Fig5.2']['2'][1], -proportions['Fig5.2']['3'][1]],  # Weak AI understanding - Local
    [-proportions['Fig5.2']['0'][2], -proportions['Fig5.2']['1'][2], -proportions['Fig5.2']['2'][2], -proportions['Fig5.2']['3'][2]],  # Moderate AI understanding - Local
    [-proportions['Fig5.2']['0'][3], -proportions['Fig5.2']['1'][3], -proportions['Fig5.2']['2'][3], -proportions['Fig5.2']['3'][3]],  # Strong AI understanding - Local
    # Global - Keeping these values positive
    [proportions['Fig6.2']['0'][1], proportions['Fig6.2']['1'][1], proportions['Fig6.2']['2'][1], proportions['Fig6.2']['3'][1]],  # Weak AI understanding - Global
    [proportions['Fig6.2']['0'][2], proportions['Fig6.2']['1'][2], proportions['Fig6.2']['2'][2], proportions['Fig6.2']['3'][2]],  # Moderate AI understanding - Global
    [proportions['Fig6.2']['0'][3], proportions['Fig6.2']['1'][3], proportions['Fig6.2']['2'][3], proportions['Fig6.2']['3'][3]],  # Strong AI understanding - Global
])

data_adjusted_t = data_adjusted.T

fig, ax = plt.subplots(figsize=(7, 4))

n_groups = data_adjusted_t.shape[0]
width = 0.4

ind = np.arange(n_groups)

p1 = ax.barh(ind, data_adjusted_t[:, 0], width, label='Acquainted_S-XAI', color = '#61B4CF')
p2 = ax.barh(ind, data_adjusted_t[:, 1], width, left=data_adjusted_t[:, 0], label='Skilled_S-XAI', color = '#3394B6')
p3 = ax.barh(ind, data_adjusted_t[:, 2], width, left=data_adjusted_t[:, 0] + data_adjusted_t[:, 1], label='Proficient_S-XAI', color = '#027399')
p4 = ax.barh(ind + width, data_adjusted_t[:, 3], width, label='Acquainted_AS-XAI', color = '#8FC1B5')
p5 = ax.barh(ind + width, data_adjusted_t[:, 4], width, left=data_adjusted_t[:, 3], label='Skilled_AS-XAI', color = '#589A8D')
p6 = ax.barh(ind + width, data_adjusted_t[:, 5], width, left=data_adjusted_t[:, 3] + data_adjusted_t[:, 4], label='Proficient_AS-XAI', color = '#007566')

ax.axvline(0, color='grey', linewidth=1.2)
ax.set_xlabel('Percentage', fontname='Arial', fontsize=10, labelpad=11)
ax.set_title('Human Evaluation of Common Semantic Space Purity', font3, pad=6)
ax.set_yticks(ind + width / 2, fontname='Arial', fontsize=10)
ax.set_yticklabels(['Not sure', 'Somewhat pure', 'Mostly pure', 'Very pure'], fontname='Arial', fontsize=10)

ax.legend(loc='upper left', prop=font)

ax.text(-0.72, -0.85, 'S-XAI', ha='right', va='center',fontname='Arial', fontsize=10, color='#016B8F', fontweight='bold')
ax.text(0.7, -0.85, 'AS-XAI', ha='left', va='center',fontname='Arial', fontsize=10, color='#007061', fontweight='bold')

ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
ax.set_xticklabels(['0.8', '0.6', '0.4', '0.2', '0', '0.2', '0.4', '0.6', '0.8'], fontname='Arial', fontsize=10)

plt.savefig(f'../lime_save_2kinds/pure.tiff',bbox_inches='tight', dpi=300)

plt.show()
'''