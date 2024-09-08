import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Creazione del DataFrame di esempio
data = {
    'VariabileNumerica1': np.random.rand(10),  # Variabile numerica casuale
    'VariabileNumerica2': np.random.randint(1, 100, 10),  # Variabile numerica casuale intera
    'VariabileCategorica': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A']  # Variabile categorica
}

df = pd.DataFrame(data)
print(df)

def anova_function(df, numeric_var, categorical_var):
    # ANOVA
    model = ols(numeric_var+ '  ~ ' + categorical_var, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Visualizzazione dei risultati dell'ANOVA
    sns.boxplot(x=categorical_var, y=numeric_var, data=df)
    plt.title('ANOVA: ' + numeric_var + " vs " + categorical_var)
    plt.show()
    print('ANOVA: ' + numeric_var + " vs " + categorical_var)
    print(anova_table)

def point_biserial_corr_function(df):
    # Coefficiente di correlazione punto seriale
    serial_corr = stats.pointbiserialr(df['VariabileNumerica2'], df['VariabileCategorica'].map({'A': 0, 'B': 1, 'C': 2}))

    # Visualizzazione del coefficiente di correlazione punto seriale
    sns.boxplot(x='VariabileCategorica', y='VariabileNumerica2', data=df)
    plt.title('VariabileNumerica2 vs VariabileCategorica')
    plt.show()
    print("Coefficiente di correlazione punto seriale:", serial_corr)

def chi_square_test_function(df):
    # Tabella di contingenza e test del chi-quadrato
    contingency_table = pd.crosstab(df['VariabileNumerica2'], df['VariabileCategorica'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)

    # Visualizzazione dei risultati del test del chi-quadrato
    sns.heatmap(contingency_table, annot=True, cmap='YlGnBu')
    plt.title('Tabella di Contingenza')
    plt.show()
    print("Valore p del test del chi-quadrato:", p)

# Eseguiamo le funzioni e visualizziamo i risultati
anova_function(df, numeric_var="VariabileNumerica1", categorical_var="VariabileCategorica")
anova_function(df, numeric_var="VariabileNumerica2", categorical_var="VariabileCategorica")

point_biserial_corr_function(df)
chi_square_test_function(df)