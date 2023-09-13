import pandas as pd
import numpy as np
from contextlib import redirect_stdout

# For local machine
# SPECIFIC_OUTPUT_DIR = "/home/bellaando/thesis23/clean_data/"
# DATA_SOURCE = "/home/bellaando/thesis23/local.csv"
# SUFFIX = "_local"

# For ARTEMIS truncated
# SPECIFIC_OUTPUT_DIR = "/project/RDS-FEI-START2-RW/clean_data/"
# DATA_SOURCE = "/project/RDS-FEI-START2-RW/TestED.csv"
# SUFFIX = "_truncated"

# For ARTEMIS big one
SPECIFIC_OUTPUT_DIR = "/project/RDS-FEI-START2-RW/clean_data/"
DATA_SOURCE = "/scratch/RDS-FEI-START2-RW/bellanew.csv"
SUFFIX = "_full"

TARGET_VAR = 'repres7days'
SELECTED_COLUMNS = ['age_recode',
                    # 'ppn', # patient identifier
                    'SEX', 
                    'ED_SOURCE_OF_REFERRAL', # where to put in employer?
                    'referred_to_on_departure_recode',
                    'PREFERRED_LANGUAGE_ASCL', 
                    'MODE_OF_ARRIVAL',
                    'MODE_OF_SEPARATION', 
                    'TRIAGE_CATEGORY', # leave as is
                    'HOURS_IN_ICU',
                    'final_diagnosis_subcode', # ask - i think this one will be more useful than ED_DIAGNOSIS CODE
                    'level', # hospital type
                    'EDLOS',
                    'remoteness', # leave as is
                    'DEATH_DATE',
                    # 'PRESENTING_PROBLEM', # keep for now, who knows
                    TARGET_VAR,
]

NEW_COLUMNS = ['age',
                # 'ppn',
                'sex',
                'source_referral',
                'departure_referral',
                'preferred_language',
                'arrival_mode',
                'separation_mode',
                'triage_category',
                'icu_status',
                'diagnosis_code',
                'level',
                'EDLOS',
                'remoteness',
                'death_status',
                # 'presenting_problem',
                TARGET_VAR,

]

ALL_COLUMNS = ['PPN',
                'referred_to_on_departure_recode',
                'age_recode',
                'arrival_date',
                'arrival_time',
                'actual_departure_date',
                'actual_departure_time',
                'PREFERRED_LANGUAGE_ASCL',
                'ED_DIAGNOSIS_CODE',
                'SEX',
                'ED_DIAGNOSIS_CODE_SCT',
                'ED_SOURCE_OF_REFERRAL',
                'MODE_OF_ARRIVAL',
                'MODE_OF_SEPARATION', 
                'TRIAGE_CATEGORY',
                'DIAGNOSIS_CODE_P',
                'HOURS_IN_ICU',
                'DEATH_DATE',
                'level',
                'EDLOS',
                'repres7days',
                'repres30days',
                'remoteness',
                'PRESENTING_PROBLEM',
                'Project_recnum',
                'Indigenous_status'
]

def count_rows_containing_nan(df):
    return df.isnull().any(axis=1).sum()

def age_to_nominal(age: float) -> int:
    if age < 6:
        return "0-5"
    elif age < 16:
        return "6-15"
    elif age < 26:
        return "16-25"
    elif age < 46:
        return "26-45"
    elif age < 66:
        return "46-65"
    elif age < 86:
        return "66-85"
    return "86+"

def sex_to_nominal(sex: int) -> str:
    if sex not in [1, 2, 3, 9]: # M, F, I, UNKNOWN respectively
        raise ValueError("Unexpected sex value")
    elif sex == 1:
        return "M"
    elif sex == 2:
        return "F"
    return "other"

def source_of_referral_to_nominal(source: int) -> str:
    if source == 1: # self, family, friends
        return "self/family/friends"
    elif source <= 4: # clinic
        return "clinic"
    elif source <= 9: # hospital
        return "hospital"
    elif source <= 16: # community org
        return "community_org"
    return "other"

def referred_to_on_departure_to_nominal(source: int) -> str:
    if source < 3: # review in ED
        return "ED_review"
    elif source == 8: # not referred
        return "no_referral"
    elif source == 9: # unknown
        return "unknown"
    return "specialist" # referred to specialist or social work

def preferred_language_ascl_to_nominal(language: int) -> str:
    if language < 1000: # unknown or nonverbal
        return "none"
    elif language == 1201: # english
        return "english"
    elif language < 4000: # european
        return "european"
    elif language < 5000: # middle eastern
        return "middle eastern"
    elif language < 8000: # asian
        return "asian"
    return "other" # catchall other

def mode_of_arrival_to_nominal(mode: int) -> str:
    if mode in [1, 4, 5, 6]: # ambulance of some sort
        return "ambulance"
    elif mode == 3: # private vehicle
        return "private_vehicle"
    return "other"

def mode_of_separation_to_nominal(mode: int) -> str:
    if mode in [1, 2, 5, 9, 10, 11, 12]: # admitted
        return "admitted"
    elif mode in [3, 8, 99]: # died or error
        return "died/other"
    elif mode in [6, 7, 13]: # left against advice
        return "left_against_advice"
    return "released" # released from ED

def hours_in_icu_to_nominal(hours: int) -> str:
    if hours > 0: # attended ICU
        return "True"
    return "False"

def final_diagnosis_subcode_to_nominal(code: float) -> str:
    return int(np.floor(code))

def ed_los_to_nominal(hours: int) -> str:
    if hours <= 4:
        return "0-4"
    elif hours <= 12:
        return "5-12"
    elif hours <= 24:
        return "13-24"
    return "25+"

def death_to_nominal(date: str) -> str:
    if date != 0:
        return "True"
    return "False"

def output_analytics(df):
    # Count number of instances containing missing values
    if df.isnull().values.any():
        n_missing = count_rows_containing_nan(df)
        n_rows = len(df)
        print(f'{n_missing} rows out of {n_rows} rows '
              f'({n_missing / n_rows * 100:.2f}%) contain missing values.\n')
        df = df.fillna('missing')
    # Cross-tabulate each attribute against class attribute
    for col in df.columns:
        if col == TARGET_VAR:
            continue
        tab = pd.crosstab(
            df[col], df[TARGET_VAR], dropna=False, margins=True)
        print(tab, '\n')

df = pd.read_csv(DATA_SOURCE, encoding='unicode_escape', names = ALL_COLUMNS)

print('Original Dataset:')
print(df.head())

df = df[SELECTED_COLUMNS]

print('Selected Columns:')
print(df.head())

# Ensure there are no NULL rows in the data
df = df.dropna(how = 'all')
assert not df.isnull().values.all()

# Coerce most columns to numeric
numeric_cols = df.columns[df.dtypes.ne('object')]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

# rename columns
df.columns = NEW_COLUMNS

print("Renamed Columns:")
print(df.head())

# Transform variables into buckets
df['age'] = df['age'].apply(age_to_nominal)
df['sex'] = df['sex'].apply(sex_to_nominal)
df['source_referral'] = df['source_referral'].apply(source_of_referral_to_nominal)
df['departure_referral'] = df['departure_referral'].apply(referred_to_on_departure_to_nominal)
df['preferred_language'] = df['preferred_language'].apply(preferred_language_ascl_to_nominal)
df['arrival_mode'] = df['arrival_mode'].apply(mode_of_arrival_to_nominal)
df['separation_mode'] = df['separation_mode'].apply(mode_of_separation_to_nominal)
df['icu_status'] = df['icu_status'].apply(hours_in_icu_to_nominal)
df['diagnosis_code'] = df['diagnosis_code'].fillna(0)
df['diagnosis_code'] = df['diagnosis_code'].apply(final_diagnosis_subcode_to_nominal)
df['death_status'] = df['death_status'].fillna(0)
df['death_status'] = df['death_status'].apply(death_to_nominal)
df['EDLOS'] = df['EDLOS'].apply(ed_los_to_nominal)

# Ensure there are no NULL rows in the data
df = df.dropna(how = 'any')
assert not df.isnull().values.any()

# convert leftovers to nominal ints
df['triage_category'] = df['triage_category'].astype(int)
df['remoteness'] = df['remoteness'].astype(int)

# get rid of corner cases; if patient died OR they had no diagnosis
df = df[(df.death_status != "True") & (df.diagnosis_code != 0)]

print("Fully Cleaned Dataset:")
print(df.head())

# export new clean data to csv
df.to_csv(SPECIFIC_OUTPUT_DIR + 'no_FS' + SUFFIX + '.csv', index=False)

# export manual FS clean data to csv
df.to_csv(SPECIFIC_OUTPUT_DIR + 'manual_FS' + SUFFIX + '.csv', index=False, columns=['age', 'icu_status', 'triage_category', 'diagnosis_code', 'remoteness', 'repres7days'])

# cross tabulate all new variables
with open(SPECIFIC_OUTPUT_DIR + 'summary' + SUFFIX + '.txt', 'w') as f:
    with redirect_stdout(f):
        output_analytics(df)