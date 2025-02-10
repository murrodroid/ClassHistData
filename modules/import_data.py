import pandas as pd
import numpy as np
import random
from modules.tools import remove_text_after_comma
from sklearn.model_selection import train_test_split

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


def import_data_individualized():
    persons = pd.read_csv(path6,sep=';', low_memory=False)

    persons = persons.assign(
        age = persons['ageYears'].fillna(0) + 
            persons['ageMonth'].fillna(0)/12 + 
            persons['ageWeeks'].fillna(0)/52 + 
            persons['ageDays'].fillna(0)/365.25 + 
            persons['ageHours'].fillna(0)/8765.81277,
        
        dateOfDeath = lambda df: pd.to_datetime(df['dateOfDeath'], errors='coerce'),
        
        sex = lambda df: df['sex'].replace('Ukendt', np.nan),
        
        deathcauses = lambda df: df['deathcauses'].astype(str).str.lower().str.strip()
    ).assign(
        yearOfDeath = lambda df: df['dateOfDeath'].dt.year,

        deathcause_mono = lambda df: df['deathcauses'].fillna('').apply(remove_text_after_comma)
    ).assign(
        icd10h = lambda df: df['deathcause_mono'].map(icd10h),
        icd10h_desc = lambda df: df['deathcauses'].map(icd10h_desc),
        dk1875 = lambda df: df['deathcauses'].map(dk1875),
        
        childcat = lambda df: df['icd10h'].map(childcat),
        infantcat = lambda df: df['icd10h'].map(infantcat),
        histcat = lambda df: df['icd10h'].map(histcat),
        heiberg = lambda df: df['dk1875'].map(heiberg)
    )

    
    df = persons[['deathcause_mono','deathcauses','age','sex','hood','yearOfDeath','icd10h','dk1875','childcat','infantcat','histcat','heiberg']]

    return df, persons

def import_data_random(retain_pct,seed=333):
    random.seed(seed)
    icd10h_random = {k: icd10h[k] for k in random.sample(list(icd10h), int(len(icd10h) * retain_pct))}
    icd10h_ordered = {k: icd10h[k] for k in list(icd10h)[:int(len(icd10h) * retain_pct)]}

    df, persons = import_data_individualized()

    df = df.assign(
        icd10h_random = lambda df: df['deathcause_mono'].map(icd10h_random),
        icd10h_ordered = lambda df: df['deathcause_mono'].map(icd10h_ordered)
    )

    return df, persons

def import_data_standard():
    """
    Imports and processes data from multiple Excel files to create a consolidated DataFrame.

    Returns:
        pd.DataFrame: DataFrame containing columns from the source files along with additional processed columns.
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

    train_df = df[df.icd10h_code.notna()]

    return train_df, df