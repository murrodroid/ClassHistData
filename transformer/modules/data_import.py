import pandas as pd
import numpy as np

path1 = './datasets/louise_icd10h_edited.xlsx'
path2 = './datasets/CHILDCAT 032024.xlsx'
path3 = './datasets/INFANTCAT 082024.xlsx'
path4 = './datasets/HISTCAT 082024.xlsx'
path5 = './datasets/heiberg.xlsx'
path6 = './datasets/KBHBegravelser_1861-1940_cleaned.csv'

icd_df = pd.read_excel(path1)
icd_df.columns = icd_df.columns.str.lower().str.replace(' ', '')
Chil = pd.read_excel(path2)
Infi = pd.read_excel(path3, sheet_name='InfantCat')
Hist = pd.read_excel(path4, sheet_name='Masterlist')
Heib = pd.read_excel(path5)

icd10h = {deathcause: icd_df['icd10h_code'][i] for i,deathcause in enumerate(icd_df['tidy_cod'])}
icd10h_desc = {deathdesc: icd_df['icd10h_description_english'][i] for i,deathdesc in enumerate(icd_df['tidy_cod'])}
dk1875 = {deathcause: icd_df['dk1875_code'][i] for i,deathcause in enumerate(icd_df['tidy_cod'])}
childcat = {icd10h: Chil['CHILDCAT'][i] for i,icd10h in enumerate(Chil['ICD10H'])}
infantcat = {icd10h: Infi['Infantcat2024'][i] for i,icd10h in enumerate(Infi['ICD10h'])}
histcat = {icd10h: Hist['HistCat'][i] for i,icd10h in enumerate(Hist['ICD10h'])}
heiberg = {dk: Heib['heiberg tbl. X'][i] for i,dk in enumerate(Heib['dk1875'])}


def import_data(target='icd10h_code'):
    """
    Imports and processes data from multiple Excel files to create DataFrames with ICD10h codes and their deathcauses.

    Returns:
        df_labeled  : pd.DataFrame, a DataFrame with labeled ICD10h codes.
        df          : pd.DataFrame, a DataFrame with all deathcauses. 
    """

    df = icd_df[['tidy_cod','icd10h_code','dk1875_code','icd10h_description_english']]
    df = df.assign(
        mono_icd10h_code=lambda df: [icd10h.get(deathcause, np.nan) for deathcause in df['tidy_cod']],
        mono_dk1875_code=lambda df: [dk1875.get(deathcause, np.nan) for deathcause in df['tidy_cod']],
        childcat_code=lambda df: [childcat.get(icd, np.nan) for icd in df['icd10h_code']],
        infantcat_code=lambda df: [infantcat.get(icd, np.nan) for icd in df['icd10h_code']],
        histcat_code=lambda df: [histcat.get(icd, np.nan) for icd in df['icd10h_code']],
        heiberg_code=lambda df: [heiberg.get(dk, np.nan) for dk in df['dk1875_code']]
    )
    
    df[['icd10h_category', 'icd10h_subcategory']] = (df['icd10h_code'].str.split('.', n=1, expand=True))
    
    if target not in df.columns:
        raise ValueError(
            f"'{target}' is not one of the dataframe columns: {list(df.columns)}"
        )

    df_labeled = df[df[target].notna()]

    return df_labeled, df

