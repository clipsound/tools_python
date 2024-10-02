import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import re
from scipy.stats import chi2_contingency, chi2
from scipy.stats import ttest_ind
from statsmodels.formula.api import ols
import statsmodels.api as sm
import seaborn as sns

sopravvivenza_y = "survival [y]"
sopravvivenza_m = "survival [m]"
metric_selected = sopravvivenza_m


def map_values(x):
    lower_x = str(x).lower()
    if lower_x.startswith('vivent'):
        return 0
    elif lower_x.startswith('deced'):
        return 1
    else:
        return -1

def calculate_kmf(dataframe):
    df = dataframe.copy()
    df['DATA CENSOR'] = pd.to_datetime(df['DATA CENSOR'], errors='coerce')
    df['DATA GK'] = pd.to_datetime(df['DATA GK'], errors='coerce')

    # computed both metrics
    df[sopravvivenza_y] = (df['DATA CENSOR'] - df['DATA GK']).dt.days / 365.25  # 365.25 giorni per anno
    df[sopravvivenza_m] = (df['DATA CENSOR'] - df['DATA GK']).dt.days / 12  # 12 mesi per anno

    df = df.dropna(subset=[metric_selected])
    kmf = KaplanMeierFitter()
    kmf.fit(durations=df[metric_selected], event_observed=df['CENSOR'])
    return kmf, df


def plot_kmf(kmf_modello):
    plt.figure(figsize=(10, 6))
    kmf_modello.plot()
    plt.xlabel('Time ' + metric_selected)
    plt.ylabel('Survival Probability')
    plt.title('Kaplan-Meier survival curve')
    plt.show()

def plot_kmf_multi(kmfs, plotname, groups_labels):
    # Plot delle curve di sopravvivenza con colori diversi
    plt.figure(figsize=(15, 10))

    for i, kmf in enumerate(kmfs):
        data_size = kmf.timeline.shape[0]
        label = f'Group ' + groups_labels[i] + '[' + str(data_size) + ']'
        kmf.plot_survival_function(label=label)

    fs = 18
    plt.xlabel('Time ' + metric_selected, fontsize=fs)
    plt.ylabel('Survival Probability', fontsize=fs)
    plt.title('Kaplan-Meier survival curve', fontsize=fs+3)
    plt.legend(loc='lower left', fontsize=fs-3)
    plt.grid(True)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    #plt.show()
    plt.savefig(plotname + '.png')

def clean_sheet_name(name):
    # Rimuovi caratteri non consentiti nei titoli dei fogli Excel
    cleaned_name = re.sub(r'[^\w\s]', '', name)
    return cleaned_name[:30]  # Limita la lunghezza del nome a 30 caratteri per la sicurezza

def stats(df):
    # Specifica i nomi delle colonne da escludere
    exclude_columns = ["NOME", "CC"]
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
    with pd.ExcelWriter('stats_dataframe.xlsx') as writer:
        numeric_stats.to_excel(writer, sheet_name='Statistiche_numeriche')

        # Creazione di una tabella per i valori unici e la loro frequenza per le colonne non numeriche
        for col in non_numeric_columns:
            col_values_counts = df[col].value_counts().reset_index()
            col_values_counts.columns = [col, 'Frequenza']
            sheet_name = clean_sheet_name(f'F_{col}')
            col_values_counts.to_excel(writer, sheet_name=sheet_name, index=False)

def data_preparation():
    # Caricamento dei dati dal foglio Excel
    percorso_file_excel = 'DATABASE_MENINGIOMI_5.xlsx'
    df = pd.read_excel(percorso_file_excel)

    # FIX DATA ------------------------------------------------------------
    datafield_to_fix = 'DATA CENSOR'
    df[datafield_to_fix] = pd.to_datetime(df[datafield_to_fix], errors='coerce')
    df = df.dropna(subset=[datafield_to_fix])

    datafield_to_fix = 'DATA GK'
    df[datafield_to_fix] = pd.to_datetime(df[datafield_to_fix], errors='coerce')
    df = df.dropna(subset=[datafield_to_fix])

    df['DATA GK'] = pd.to_datetime(df['DATA GK'])
    df['DATA CENSOR'] = pd.to_datetime(df['DATA CENSOR'])

    # SETUP CENSOR ------------------------------------------------------
    field = "CENSOR"
    df[field] = df[field].apply(map_values)
    df = df[df[field] != 'Perso al FU']




    return df

def km_Generic2class(dataframe, filterName=None, list_of_value1=None, list_of_value2=None, perfix_image_save=""):
    groups_labels = []
    groups_labels.append("All patients")
    km_s = []
    km0, df0n = calculate_kmf(dataframe)
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

def km_Generic3class(dataframe, filterName=None, list_of_value1=None, list_of_value2=None, list_of_value3=None, perfix_image_save=""):
    groups_labels = []
    km_s = []
    if filterName is not None:
        groups_labels.append(filterName + ":" + '|'.join(list_of_value1))
        groups_labels.append(filterName + ":" + '|'.join(list_of_value2))
        groups_labels.append(filterName + ":" + '|'.join(list_of_value3))
        df1 = dataframe.loc[dataframe[filterName].str.lower().isin(map(str.lower, list_of_value1))]
        df2 = dataframe.loc[dataframe[filterName].str.lower().isin(map(str.lower, list_of_value2))]
        df3 = dataframe.loc[dataframe[filterName].str.lower().isin(map(str.lower, list_of_value3))]
        km1, df1n = calculate_kmf(df1)
        km2, df2n = calculate_kmf(df2)
        km3, df3n = calculate_kmf(df3)
        km_s.append(km1)
        km_s.append(km2)
        km_s.append(km3)
        plot_kmf_multi(kmfs=km_s, plotname=(perfix_image_save + filterName), groups_labels=groups_labels)
        return df1, df2, df3
    else:
        return None, None, None


def filter_atomic(df, f1):
    df1 = df.loc[df[f1.filter_name].isin(f1.list_of_values_group1)]
    df2 = df.loc[df[f1.filter_name].isin(f1.list_of_values_group1)]
    return df1, df2

def anova_function(df, numeric_var, categorical_var, save_xls = False):
    # ANOVA
    model = ols(f"{numeric_var} ~ {categorical_var}", data=df).fit()


    anova_table = sm.stats.anova_lm(model, typ=2)
    sns.boxplot(x=categorical_var, y=numeric_var, data=df)
    labels = df[categorical_var].unique()

    # Associa ogni label a un handle
    handles = [plt.Line2D([], [], color=sns.color_palette()[i], label=label) for i, label in enumerate(labels)]
    plt.legend(handles=handles)

    plotname = 'ANOVA ' + numeric_var + " vs " + categorical_var
    plt.title(plotname)
    filename = plotname.replace(' ', '_')
    plt.savefig(filename + '.png')
    if save_xls:
        with pd.ExcelWriter(filename + '.xlsx') as writer:
            anova_table.to_excel(writer, sheet_name=plotname)

def anova(df, categorical_val = 'FU NeuroRx'):

    conditions = [
        (df[categorical_val].isin(['RIDOTTO', 'SCOMPARSO', 'RIDOTTO dal 50% al 90%', 'INVARIATO'])),
        (df[categorical_val].isin(['AUMENTATO', 'REDICIVA NL.', 'NON NOTO']))
    ]
    yes_val = 'Responders'
    no_val = 'No Responders'
    values = [yes_val, no_val]
    new_column = categorical_val.replace(' ', '_') + "_2C"
    reference_column = (new_column)  # 'FU CLINICO'
    df[new_column] = np.select(conditions, values, default=no_val)

    df = df.rename(columns=lambda x: x.replace(' ', '_'))
    df = df.rename(columns=lambda x: x.replace('°', '_'))
    df = df.rename(columns=lambda x: x.replace('%', 'perc'))

    colonne_da_confrontare = ["ETA", 'VOLUME_in_cc', 'IPE_perc', 'DS_Gy', 'DM_Gy', 'N__SHOTS', 'AV_DOSE_Gy', 'VOL_12_Gy']

    for colonna in colonne_da_confrontare:
        null_values = df[df[colonna].isnull()]
        non_numeric_values = df[pd.to_numeric(df[colonna], errors='coerce').isnull()]
        if len(non_numeric_values) > 0 or len(null_values) > 0:
            df = df.dropna(subset=[colonna], how='any')
            df = df[pd.to_numeric(df[colonna], errors='coerce').notnull()]
        num_rows = df.shape[0]  # df è il tuo DataFrame
        print(f"Numero di righe del DataFrame: {num_rows}")
        anova_function(df,colonna,reference_column)



def chi_square(df, categorical_val='FU NeuroRx'):
    conditions = [
        (df[categorical_val].isin(['RIDOTTO', 'SCOMPARSO', 'RIDOTTO dal 50% al 90%', 'INVARIATO'])),
        (df[categorical_val].isin(['AUMENTATO', 'REDICIVA NL.', 'NON NOTO']))
    ]
    yes_val = 'Responders'
    no_val = 'No Responders'
    values = [yes_val, no_val]
    new_column = categorical_val.replace(' ', '_') + "_2C"
    df[new_column] = np.select(conditions, values, default=no_val)
    column_to_compare = (new_column)
    columns_to_exclude = [column_to_compare, 'CENSOR']
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in columns_to_exclude]

    # Eseguire il test del chi-quadro per ciascuna variabile numerica rispetto alla colonna categorica

    results = []

    for numeric_column in numeric_columns:
        contingency_table = pd.crosstab(df[column_to_compare], df[numeric_column])
        chi2_stats, p_value, _, _ = chi2_contingency(contingency_table)

        result_dict = {
            'Column_Compare': column_to_compare,
            'Numeric_Column': numeric_column,
            'Chi_Square_Value': chi2_stats,
            'P_Value': p_value
        }
        results.append(result_dict)

        print(f"Chi-square test for '{column_to_compare}' vs '{numeric_column}':")
        print(f"Chi-square value: {chi2_stats}")
        print(f"P-value: {p_value}")
        print("-----------------------------------------------------")
        if p_value < 0.05:
            degrees_of_freedom = (contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1)

            x = np.linspace(0, 50, 1000)
            chi2_dist = chi2(degrees_of_freedom)
            y = chi2_dist.pdf(x)
            plt.clf()
            plt.plot(x, y, label=f'Degrees of Freedom: {degrees_of_freedom}')
            plt.axvline(x=chi2_stats, color='red', linestyle='--',
                        label=f'Chi-Squared: {chi2_stats:.2f}, p-value: {p_value:.4f}')
            plt.legend()
            plt.xlabel('Chi-Squared Values')
            plt.ylabel('Probability Density Function')
            plotname=f"Chi-Squared Distribution ['{numeric_column} vs {column_to_compare}]"
            plt.title(plotname)
            filename = plotname.replace(' ', '_')
            plt.savefig(filename + '.png')
    results_df = pd.DataFrame(results)
    results_df.to_excel('chi_square_results.xlsx', index=False)

def ttest(df):
    conditions = [
        (df['FU NeuroRx'].isin(['RIDOTTO', 'SCOMPARSO', 'RIDOTTO dal 50% al 90%', 'INVARIATO'])),
        (df['FU NeuroRx'].isin(['AUMENTATO', 'REDICIVA NL.', 'NON NOTO']))
    ]
    yes_val = 'Responders'
    no_val = 'No Responders'
    values = [yes_val, no_val]
    reference_column = ('FU NeuroRx_NEW')  # 'FU CLINICO'
    df['FU NeuroRx_NEW'] = np.select(conditions, values, default='Altro')
    colonne_da_confrontare = [ "ETA",  'VOLUME in cc','IPE %', 'DS Gy', 'DM Gy', 'N° SHOTS', 'AV DOSE Gy', 'VOL 12 Gy']

    for colonna in colonne_da_confrontare:
        null_values = df[df[colonna].isnull()]
        non_numeric_values = df[pd.to_numeric(df[colonna], errors='coerce').isnull()]
        if len(non_numeric_values) > 0 or len(null_values) > 0:
            file_prefix = 'ERROR_' + colonna + '.xlsx'
            with pd.ExcelWriter(file_prefix) as writer:
                null_values.to_excel(writer, sheet_name='Null Values')
                non_numeric_values.to_excel(writer, sheet_name='Non Numeric Values')
            df = df.dropna(subset=[colonna], how='any')
            df = df[pd.to_numeric(df[colonna], errors='coerce').notnull()]


        gruppo_yes = df[df[reference_column] == yes_val][colonna]
        gruppo_no = df[df[reference_column] == no_val][colonna]

        # Esegui il test t di Student per confrontare le medie delle colonne tra i due gruppi
        t_stat, p_val = ttest_ind(gruppo_yes, gruppo_no,
                                  equal_var=False)  # equal_var=False se le varianze non sono uguali

        # Stampa i risultati
        print(f"Test t per '{colonna}':")
        print(f"Statistiche t: {t_stat}")
        print(f"Valore p: {p_val}")
        if p_val < 0.05:  # Considera un livello di significatività del 5%
            print("La differenza è significativa.\n")
        else:
            print("La differenza non è significativa.\n")


def parson(df):
    #not working
    conditions = [
        (df['FU NeuroRx'].isin(['RIDOTTO', 'SCOMPARSO', 'RIDOTTO dal 50% al 90%', 'INVARIATO'])),
        (df['FU NeuroRx'].isin(['AUMENTATO', 'REDICIVA NL.', 'NON NOTO']))
    ]
    yes_val = 'Responders'
    no_val = 'No Responders'

    values = [yes_val, no_val]

    reference_column = ('FU NeuroRx_tmp')  # 'FU CLINICO'
    reference_column2 = ('FU NeuroRx_NEW')
    df[reference_column] = np.select(conditions, values, default='Altro')

    # Converti 'FU NeuroRx' in variabili dummy
    dummy_columns = pd.get_dummies(df[reference_column])

    # Somma delle colonne binarie per ottenere una singola colonna
    df[reference_column2] = dummy_columns.sum(axis=1)

    # Calcola la correlazione di Pearson tra la nuova colonna e le altre colonne del DataFrame
    correlation = df.corrwith(df[reference_column2])

    correlation.plot(kind='bar', figsize=(12, 6))
    plt.title('Correlazione tra FU NeuroRx e altre colonne')
    plt.xlabel('Variabili')
    plt.ylabel('Correlazione di Pearson')
    plt.show()



def kmplots(df):
    km_Generic2class(df, perfix_image_save="KMF_")
    dfPRE, dfNOPRE = km_Generic2class(df, filterName="TRATT Pre GK", list_of_value1=['SI', 'SI > 1'], list_of_value2=['NO'], perfix_image_save="KMF_")
    dfSUB, dfGROSS = km_Generic2class(df, filterName="ESITO TRATTAMENTO", list_of_value1=['subtotal resection'], list_of_value2=['Gross-total resection'], perfix_image_save="KMF_")
    dfPRE_SUB, dfPRE_GROSS = km_Generic2class(dfPRE, filterName="ESITO TRATTAMENTO", list_of_value1=['subtotal resection'], list_of_value2=['Gross-total resection'], perfix_image_save="KMF_PREGKYES_")


    dfVOLTA, dfOther = km_Generic2class(df, filterName="SEDE", list_of_value1=['Volta'], list_of_value2=['central skull base', 'FC posteriore', 'Altro', 'FC anteriore'], perfix_image_save="KMF_")
    dfPRE_VOLTA, dfPRE_Other = km_Generic2class(dfPRE, filterName="SEDE", list_of_value1=['Volta'], list_of_value2=['central skull base', 'FC posteriore', 'Altro', 'FC anteriore'], perfix_image_save="KMF_PREGKYES_")

    dfG1, dfG2 = km_Generic2class(df, filterName="ISTOLOGIA", list_of_value1=['G1', 'XX'], list_of_value2=['G2'], perfix_image_save="KMF_")
    dfPRE_G1, dfPRE_G2 = km_Generic2class(dfPRE, filterName="ISTOLOGIA", list_of_value1=['G1', 'XX'], list_of_value2=['G2'], perfix_image_save="KMF_PREGKYES_")
    dfG1b, dfG2b, dfUNKNOWN= km_Generic3class(df, filterName="ISTOLOGIA", list_of_value1=['G1', 'XX'], list_of_value2=['G2'], list_of_value3=['unknown'], perfix_image_save="KMF_3WAY_")




df = data_preparation()
stats(df)
chi_square(df.copy(), categorical_val='FU NeuroRx')
#chi_square(df.copy(), categorical_val='FU CLINICO' )
anova(df.copy(), categorical_val='FU NeuroRx')
#anova(df.copy(), categorical_val='FU CLINICO')
kmplots(df)


#ttest(df.copy())
#parson(df.copy())
