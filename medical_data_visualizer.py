import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2 - Add an overweight column to the data based on the BMI calculation
# BMI = weight (kg) / (height (m))^2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)) > 25
df['overweight'] = df['overweight'].astype(int)

# 3 - Normalize the cholesterol and gluc columns (0 = normal, 1 = above normal)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4 - Function to draw categorical plot
def draw_cat_plot():
    # 5 - Create a DataFrame for the cat plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group and reformat the data for the catplot
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='count')

    # 7 - Draw the categorical plot using seaborn's catplot
    fig = sns.catplot(x='variable', hue='value', col='cardio', data=df_cat, kind='count')

    # 8 - Get the figure for output
    fig = fig.fig

    # 9 - Save the catplot figure
    fig.savefig('catplot.png')
    return fig

# 10 - Function to draw heatmap
def draw_heat_map():
    # 11 - Clean the data by filtering out incorrect data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure should be lower or equal to systolic
        (df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(0.975)) &  # Height within range
        (df['weight'] >= df['weight'].quantile(0.025)) & (df['weight'] <= df['weight'].quantile(0.975))  # Weight within range
    ]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle of the heatmap
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 - Plot the correlation matrix using seaborn's heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', cmap='coolwarm', ax=ax, cbar_kws={'shrink': .8})

    # 16 - Save the heatmap figure
    fig.savefig('heatmap.png')
    return fig
