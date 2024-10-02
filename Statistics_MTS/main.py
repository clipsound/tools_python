import pandas as pd
from reportlab import *
from pathlib import Path
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import re
from scipy.stats import chi2_contingency, chi2
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
import numpy as np
import re


fup_y = "FUPy"
fup_m = "FUPm"
metric_selected = fup_m
def map_values(x):
    lower_x = str(x).lower()
    if lower_x.startswith('Vivente'):
        return 0
    elif lower_x.startswith('Deceduto'):
        return 1
    else:
        return -1

def map_values2(x): #local tumor progression free survival
    lower_x = str(x).lower()
    if lower_x.startswith('NO') or x.startswith('NO'): #non si è verificato l' evento della recidiva locale
        return 0
    elif lower_x.startswith('SI') or x.startswith('SI'): #si è verificato l'evento della recidiva locale
        return 1
    else:
        return -1


def prepare_data_stats_evaluation(data, boolean_col, numeric_col1, numeric_col2, threshold1, threshold2):
    df = pd.DataFrame(data)
    print("Dati iniziali:", df.head())

    # Filtriamo i dati
    filtered_df = df[(df[boolean_col] == 1) &
                     (df[numeric_col1] >= threshold1) &
                     (df[numeric_col2] <= threshold2)]

    # Verifica il DataFrame filtrato
    print("Dimensioni del DataFrame filtrato:", filtered_df.shape)
    print("Contenuto del DataFrame filtrato:\n", filtered_df.head())
    return filtered_df


def chi_square_test(df, boolean_col, numeric_col1):
    print("Colonne nel DataFrame:", df.head(5))  # Aggiungi questo per il debug
    if boolean_col not in df.columns or numeric_col1 not in df.columns:
        raise ValueError(f"Colonne {boolean_col} o {numeric_col1} non trovate nel DataFrame.")

    # Creiamo una tabella di contingenza
    contingency_table = pd.crosstab(df[boolean_col], df[numeric_col1])
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)


    return chi2_stat, p_value


def fisher_exact_test(df, boolean_col, numeric_col1):
    """
    Esegue un test esatto di Fisher su un DataFrame.

    Parametri:
    df (pd.DataFrame): Il DataFrame da analizzare.
    boolean_col (str): Nome della colonna booleana (0 o 1).
    numeric_col1 (str): Nome della colonna numerica per la tabella di contingenza.

    Ritorna:
    Odds Ratio e p-value.
    """

    # Creiamo una tabella di contingenza
    contingency_table = pd.crosstab(df[boolean_col], df[numeric_col1])
    oddsratio, p_value = stats.fisher_exact(contingency_table)

    return oddsratio, p_value


def proportion_test(successes, totals):
    """
    Esegue un test per proporzioni.

    Parametri:
    successes (list): Numero di successi per ciascun gruppo.
    totals (list): Totale di osservazioni per ciascun gruppo.

    Ritorna:
    Z-statistic e p-value.
    """

    z_stat, p_value = sm.stats.proportions_ztest(successes, totals)

    return z_stat, p_value

def data_preparation(percorso_file_excel):

    df = pd.read_excel(percorso_file_excel)

    # FIX DATA ------------------------------------------------------------
    datafield_to_fix = 'DATA CENSOR'
    df[datafield_to_fix] = pd.to_datetime(df[datafield_to_fix], errors='coerce')
    df = df.dropna(subset=[datafield_to_fix])

    datafield_to_fix = 'DATA GK'
    df[datafield_to_fix] = pd.to_datetime(df[datafield_to_fix], errors='coerce')
    df = df.dropna(subset=[datafield_to_fix])


    # computed both metrics
    df[fup_y] = (df['DATA CENSOR'] - df['DATA GK']).dt.days / 365.25  # 365.25 giorni per anno
    df[fup_m] = (df['DATA CENSOR'] - df['DATA GK']).dt.days / 30  # 12 mesi per anno

    print(df[['FUPm', 'FUP mesi']].head(10))



    # SETUP CENSOR ------------------------------------------------------
    field = "CENSOR"
    df[field] = df[field].apply(map_values)
    df = df[df[field] != 'Perso al FU']


    # ADJUST DATA ------------------------------------------------------
    df[['Cognome', 'Nome', 'Codice']] = df['Cognome Nome'].str.split(' ', n=2, expand=True)
    df = df.drop(columns=['Cognome Nome'])

    return df

def clean_sheet_name(name):
    # Rimuovi caratteri non consentiti nei titoli dei fogli Excel
    cleaned_name = re.sub(r'[^\w\s]', '', name)
    return cleaned_name[:30]  # Limita la lunghezza del nome a 30 caratteri per la sicurezza


def statistics(dataframe, filename=""):
    # Specifica i nomi delle colonne da escludere
    df= dataframe
    exclude_columns = ["Cognome", "Nome", "Codice","CC"]
    exclude_columns += [col for col in df.columns if col.startswith("DATA")]

    # Estrazione delle colonne numeriche e non numeriche, escludendo quelle specificate
    numeric_columns = df.select_dtypes(include=np.number).columns.difference(exclude_columns).tolist()
    non_numeric_columns = df.select_dtypes(exclude=np.number).columns.difference(exclude_columns).tolist()

    # Approssimazione dei valori numerici decimali a tre cifre decimali
    for col in numeric_columns:
        df[col] = df[col].round(3)

    # Calcolo delle statistiche di primo ordine per le colonne numeriche
    numeric_stats = df[numeric_columns].describe().transpose()

    # Creazione di un file Excel con statistiche di primo ordine per colonne numeriche

    nome_file = filename.rsplit('.', 1)[0]
    if len(nome_file)>0:
        nome_file += "_"
    with pd.ExcelWriter(nome_file + 'stats_dataframe.xlsx') as writer:
        numeric_stats.to_excel(writer, sheet_name='Statistiche_numeriche')

        # Creazione di una tabella per i valori unici e la loro frequenza per le colonne non numeriche
        for col in non_numeric_columns:
            col_values_counts = df[col].value_counts().reset_index()
            col_values_counts.columns = [col, 'Frequenza']
            sheet_name = clean_sheet_name(f'F_{col}')
            col_values_counts.to_excel(writer, sheet_name=sheet_name, index=False)

def plot_kmf_multi(kmfs, plotname, groups_labels, title='Kaplan-Meier survival curve'):
    # Plot delle curve di sopravvivenza con colori diversi
    plt.figure(figsize=(15, 10))

    for i, kmf in enumerate(kmfs):
        data_size = kmf.timeline.shape[0]
        label = f'Group ' + groups_labels[i] + '[' + str(data_size) + ']'
        kmf.plot_survival_function(label=label)

    fs = 18
    plt.xlabel('Time ' + metric_selected, fontsize=fs)
    plt.ylabel('Probability', fontsize=fs)
    plt.title(title, fontsize=fs+3)
    plt.legend(loc='lower left', fontsize=fs-3)
    plt.grid(True)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    #plt.show()
    plt.savefig(plotname + '.png')

def calculate_kmf(dataframe, censor='CENSOR'):
    df = dataframe.copy()

    df = df.dropna(subset=[metric_selected])
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[metric_selected], event_observed=df[censor])
    return kmf, df



def km_Generic2class_with_numerical_filter(dataframe,
                                           column1=None,
                                           condition1=None,
                                           column2=None,
                                           condition2=None,
                                           perfix_image_save="",
                                           censor_name='CENSOR'):




    groups_labels = []
    groups_labels.append("All patients")
    km_s = []
    km0, df0n = calculate_kmf(dataframe, censor=censor_name)
    km_s.append(km0)



    # Costruisci maschere booleane basate sulle condizioni fornite
    mask1 = condition1(df[column1]) if column1 and condition1 else pd.Series([True] * len(df))
    mask2 = condition2(df[column2]) if column2 and condition2 else pd.Series([True] * len(df))

    # Filtra il DataFrame in base alle maschere create
    df_true_true = df.loc[mask1 & mask2]
    print(len(df_true_true))
    df_true_false = df.loc[mask1 & ~mask2]
    print(len(df_true_false))
    df_false_true = df.loc[~mask1 & mask2]
    df_false_false = df.loc[~mask1 & ~mask2]


    groups_labels.append("FupM >= 12 months and GK-CH > 30 days")
    groups_labels.append("FupM >= 12 months and GK-CH <= 30 days")

    km1, df1n = calculate_kmf(df_true_true, censor=censor_name)
    km2, df2n = calculate_kmf(df_true_false, censor=censor_name)
    km_s.append(km1)
    km_s.append(km2)
    plot_kmf_multi(kmfs=km_s, plotname=(perfix_image_save + column2), groups_labels=groups_labels, title="Kaplan-Meier Local Tumor Progression Free curve")

    return None, None


def km_Generic2class(dataframe, filterName=None, list_of_value1=None, list_of_value2=None, perfix_image_save=""):
    groups_labels = []
    groups_labels.append("All patients")
    km_s = []
    km0, df0n = calculate_kmf(dataframe, censor='CENSOR')
    km_s.append(km0)

    if filterName is not None:
        groups_labels.append(filterName + ":" + '|'.join(list_of_value1))
        groups_labels.append(filterName + ":" + '|'.join(list_of_value2))
        df1 = dataframe.loc[dataframe[filterName].str.lower().isin(map(str.lower, list_of_value1))]
        df2 = dataframe.loc[dataframe[filterName].str.lower().isin(map(str.lower, list_of_value2))]
        km1, df1n = calculate_kmf(df1)
        km2, df2n = calculate_kmf(df2)
        km_s.append(km1)
        km_s.append(km2)
        plot_kmf_multi(kmfs=km_s, plotname=(perfix_image_save + filterName), groups_labels=groups_labels)
        return df1, df2
    else:
        filterName = "ALL PATIENTS"
        plot_kmf_multi(kmfs=km_s, plotname=(perfix_image_save + filterName), groups_labels=groups_labels)
        return None, None


if __name__ == "__main__":
    file ="MTS_CH_GK_def3.xlsx"
    df = data_preparation(percorso_file_excel=file)
    statistics(dataframe=df, filename=file)


    # km_Generic2class(df, perfix_image_save="KMF_")
    _, _ = km_Generic2class(df, filterName="LOCAL TUMOR PROGR RECURR", list_of_value1=['SI'], list_of_value2=['NO'], perfix_image_save="SURVIVE_KMF_")
    _, _ = km_Generic2class(df, filterName="A DISTANZA (cerebrale e body)", list_of_value1=['SI'], list_of_value2=['NO'], perfix_image_save="SURVIVE_KMF_")
    _, _ = km_Generic2class(df, filterName="PRIMITIVO per meta", list_of_value1=['POLMONE'], list_of_value2=['MAMMELLA','TUBO DIGER','RENE','OVAIO','MELANOMA','ALTRO'], perfix_image_save="SURVIVE_KMF1_")
    _, _ = km_Generic2class(df, filterName="PRIMITIVO per meta", list_of_value1=['MAMMELLA'], list_of_value2=['POLMONE','TUBO DIGER','RENE','OVAIO','MELANOMA','ALTRO'], perfix_image_save="SURVIVE_KMF2_")
    _, _ = km_Generic2class(df, filterName="LATO", list_of_value1=['sx'], list_of_value2=['dx'], perfix_image_save="SURVIVE_KMF_")
    _, _ = km_Generic2class(df, filterName="SEDE", list_of_value1=['frontale'], list_of_value2=['cerebellare', 'occipitale', 'parietale', 'temporale'], perfix_image_save="SURVIVE_KMF1_")
    _, _ = km_Generic2class(df, filterName="SEDE", list_of_value1=['cerebellare'], list_of_value2=['frontale', 'occipitale', 'parietale', 'temporale'], perfix_image_save="SURVIVE_KMF2_")
    _, _ = km_Generic2class(df, filterName="Sesso", list_of_value1=['M'], list_of_value2=['F'], perfix_image_save="SURVIVE_KMF_")

    condition1 = lambda col: col >= 12 #Fup mesi >=12
    condition2 = lambda col: col <= 30  #GK days<=30

    field2 = "LOCAL TUMOR PROGR RECURR"
    df[field2] = df[field2].apply(map_values2)


    km_Generic2class_with_numerical_filter(df,
                                           column1=fup_m,
                                           condition1=condition1,
                                           column2='GIORNI CH-GK',
                                           condition2=condition2,
                                           perfix_image_save="LOCALFREETUMOR_KMF_",
                                           censor_name=field2)









    # Prepariamo i dati
    var_to_compare = 'GIORNI CH-GK'
    filtered_df = prepare_data_stats_evaluation(df, field2, fup_m, var_to_compare, 12, 30)

    # Eseguiamo il test del Chi-Quadro
    chi2_stat, p_value = chi_square_test(filtered_df, field2, var_to_compare)
    print(f'Chi-Quadro statistic: {chi2_stat}, P-value: {p_value}')

    # Eseguiamo il test esatto di Fisher
    oddsratio, p_value_fisher = fisher_exact_test(filtered_df, field2, var_to_compare)
    print(f'Odds Ratio: {oddsratio}, P-value (Fisher): {p_value_fisher}')


