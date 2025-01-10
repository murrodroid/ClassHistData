import pandas as pd
import numpy as np
from modules.tools import remove_text_after_comma

path1 = './datasets/louise_icd10h_edited.xlsx'
path2 = './datasets/CHILDCAT 032024.xlsx'
path3 = './datasets/INFANTCAT 082024.xlsx'
path4 = './datasets/HISTCAT 082024.xlsx'
path5 = './datasets/heiberg.xlsx'
path6 = './datasets/KBHBegravelser_1861-1940_cleaned.csv'


def import_data_individualized():
    persons = pd.read_csv(path6,sep='$')
    persons['age'] = persons['ageYears'].fillna(0) + persons['ageMonth'].fillna(0)/12 + persons['ageWeeks'].fillna(0)/52 + persons['ageDays'].fillna(0)/365.25 + persons['ageHours'].fillna(0)/8765.81277
    persons['dateOfDeath'] = pd.to_datetime(persons['dateOfDeath'])
    persons['yearOfDeath'] = persons['dateOfDeath'].dt.year
    persons['sex'] = [np.nan if x == 'Ukendt' else x for x in persons['sex']]
    persons['deathcauses'] = persons['deathcauses'].astype(str).str.lower()
    persons['deathcause_mono'] = persons['deathcauses'].apply(remove_text_after_comma)





def import_data_standard():
    """
    Imports and processes data from multiple Excel files to create a consolidated DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing columns from the source files along with additional processed columns.
    """
    icd_df = pd.read_excel(path1)
    icd_df.columns = icd_df.columns.str.lower().str.replace(' ', '')

    Child = pd.read_excel(path2)
    Inf = pd.read_excel(path3, sheet_name='InfantCat')
    Hist = pd.read_excel(path4, sheet_name='Masterlist')
    Heib = pd.read_excel(path5)

    icd10h = {deathcause: icd_df['icd10h_code'][i] for i,deathcause in enumerate(icd_df['tidy_cod'])}
    icd10h_desc = {deathdesc: icd_df['icd10h_description_english'][i] for i,deathdesc in enumerate(icd_df['tidy_cod'])}
    dk1875 = {deathcause: icd_df['dk1875_code'][i] for i,deathcause in enumerate(icd_df['tidy_cod'])}
    childcat = {icd10h: Child['CHILDCAT'][i] for i,icd10h in enumerate(Child['ICD10H'])}
    infantcat = {icd10h: Inf['Infantcat2024'][i] for i,icd10h in enumerate(Inf['ICD10h'])}
    histcat = {icd10h: Hist['HistCat'][i] for i,icd10h in enumerate(Hist['ICD10h'])}
    heiberg = {dk: Heib['heiberg tbl. X'][i] for i,dk in enumerate(Heib['dk1875'])}

    df = icd_df[['tidy_cod','icd10h_code','dk1875_code','icd10h_description_english']]
    df = df.assign(
        mono_icd10h_code=lambda df: [icd10h.get(deathcause, np.nan) for deathcause in df['tidy_cod']],
        mono_dk1875_code=lambda df: [dk1875.get(deathcause, np.nan) for deathcause in df['tidy_cod']],
        childcat_code=lambda df: [childcat.get(icd, np.nan) for icd in df['icd10h_code']],
        infantcat_code=lambda df: [infantcat.get(icd, np.nan) for icd in df['icd10h_code']],
        histcat_code=lambda df: [histcat.get(icd, np.nan) for icd in df['icd10h_code']],
        heiberg_code=lambda df: [heiberg.get(dk, np.nan) for dk in df['dk1875_code']]
    )

    train_df = df[df.icd10h_code.notna()]

    return train_df, df