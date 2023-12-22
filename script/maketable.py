#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import six

yourpath = "path/to/your/"

#Need to change the file for different table (anxiet/depression, original/perturbation)
infile_path = os.path.join(yourpath, 'Five_model_comparison_results_anxiety_perturbation_data.csv')
table_path = os.path.join(yourpath, 'Five_model_comparison_results_anxiety_perturbation_data_table.pdf')

    

df= pd.read_csv(infile_path)

# Prepare a new DataFrame to store the combined mean and std values
combined_df = pd.DataFrame()
combined_df['Model'] = df['Model']

# Iterate over columns to combine mean and std values
for col in df.columns:
    if col.startswith('Mean '):
        # Extract the metric name without ' Mean' or ' Std'
        metric_base = col.replace('Mean ', '')
        # Combine mean and std into a single string
        combined_df[str(metric_base)] = round(df[f'Mean {metric_base}'],2).astype(str) + " ± " + round(df[f'Std {metric_base}'],2).astype(str)

combined_df.drop(columns=['roc_auc', 'Composite Score'], inplace=True)
        
    
# Plotting the table
fig, ax = plt.subplots(figsize=(12, 5))  # set size frame
#ax.axis('tight')
ax.axis('off')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24

table = plt.table(cellText=combined_df.values, 
                  colLabels=combined_df.columns, 
                  cellLoc='center', 
                  loc='center',
                  fontsize=24,
                  colColours=["skyblue"] * df.shape[1])

table.scale(1.2, 3)
plt.title("Anxiety (Perturbed Data)", fontsize=18)

plt.savefig(table_path)
plt.show()

