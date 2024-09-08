import pandas as pd
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, PageBreak, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import cm

import numpy as np
import re


def map_values(x):
    lower_x = str(x).lower()
    if lower_x.startswith('Vivente'):
        return 0
    elif lower_x.startswith('Deceduto'):
        return 1
    else:
        return -1


def data_preparation(percorso_file_excel):
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

    # ADJUST DATA ------------------------------------------------------
    df[['Cognome', 'Nome', 'Codice']] = df['Cognome Nome'].str.split(' ', n=2, expand=True)
    df = df.drop(columns=['Cognome Nome'])

    return df

def clean_sheet_name(name):
    # Rimuovi caratteri non consentiti nei titoli dei fogli Excel
    cleaned_name = re.sub(r'[^\w\s]', '', name)
    return cleaned_name[:30]  # Limita la lunghezza del nome a 30 caratteri per la sicurezza


def stats(df):
    # Specifica i nomi delle colonne da escludere
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
    with pd.ExcelWriter('stats_dataframe.xlsx') as writer:
        numeric_stats.to_excel(writer, sheet_name='Statistiche_numeriche')

        # Creazione di una tabella per i valori unici e la loro frequenza per le colonne non numeriche
        for col in non_numeric_columns:
            col_values_counts = df[col].value_counts().reset_index()
            col_values_counts.columns = [col, 'Frequenza']
            sheet_name = clean_sheet_name(f'F_{col}')
            col_values_counts.to_excel(writer, sheet_name=sheet_name, index=False)






df = data_preparation(percorso_file_excel="Statistiche_MTS_CH_GK def2.xlsx")
stats(df)

def create_pdf(dataframe):
    # Creiamo il PDF
    pdf_filename = "output_reportlab.pdf"
    pdf = SimpleDocTemplate(
        pdf_filename,
        pagesize=landscape(A4),
        topMargin=0.5 * cm,  # Margine superiore
        bottomMargin=0.5 * cm,  # Margine inferiore
        leftMargin=0.5 * cm,  # Margine sinistro
        rightMargin=0.5 * cm  # Margine destro
    )
    def chunk_dataframe(dataframe, chunk_size):
        for i in range(0, len(dataframe), chunk_size):
            yield dataframe.iloc[i:i + chunk_size]

    styles = getSampleStyleSheet()
    style_normal = styles["Normal"]
    style_normal.fontSize = 8  # Riduciamo la dimensione del carattere
    style_normal.alignment = TA_CENTER

    chunk_size = 6  # Puoi adattare il numero di righe per ogni pagina
    elementi = []

    for chunk in chunk_dataframe(dataframe, chunk_size):
        # Convertiamo i dati in liste di liste, usando Paragraph per le celle lunghe
        dati_tabella = [chunk.columns.tolist()] + [
            [Paragraph(str(cell), style_normal) if isinstance(cell, str) else cell for cell in row]
            for row in chunk.values.tolist()
        ]

        tabella = Table(dati_tabella)

        # Stile della tabella
        stile = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),  # Font ridotto per la tabella
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])

        tabella.setStyle(stile)

        # Aggiungiamo la tabella alla lista degli elementi
        elementi.append(tabella)
        elementi.append(PageBreak())  # Aggiungiamo un salto pagina

    # Costruzione del PDF
    pdf.build(elementi)