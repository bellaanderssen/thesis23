import pandas as pd

TARGET_VAR = 'repres7days'
SELECTED_COLUMNS = ['age_recode', 'referred_to_on_departure_recode',
                    'PREFERRED_LANGUAGE_ASCL', 'ED_DIAGNOSIS_CODE',
                    'SEX', 'ED_DIAGNOSIS_CODE_SCT', 'MODE_OF_ARRIVAL',
                    'MODE_OF_SEPARATION', 'TRIAGE_CATEGORY', 'ED_STATUS',
                    'DIAGNOSIS_CODE_P', 'EPISODE_LENGTH_OF_STAY', 'HOURS_IN_ICU',
                    'MARITAL_STATUS', 'MDC', 'UNIT_TYPE_ON_ADMISSION',
                    'DEATH_DATE', 'final_diagnosis_subcode', 'level', 'EDLOS', 'remoteness',
                    'PRESENTING_PROBLEM', TARGET_VAR,
                    ]

def count_rows_containing_nan(df):
    return df.isnull().any(axis=1).sum()


def sex_to_nominal(sex: int) -> int:
    if sex not in [1, 2, 3, 9]:
        raise ValueError("Unexpected sex value")
    if sex > 3:
        return 3
    return sex

def age_to_nominal(age: float) -> int:
    if age < 20:
        return 0
    elif age < 40:
        return 1
    elif age < 60:
        return 2
    elif age < 80:
        return 3
    return 4

def source_of_referral_to_nominal(source: int) -> int:
    if source < 0:
        raise ValueError("Source of referral cannot be negative")
    if source in [1]:
        return 1
    elif source in [2, 3, 6, 7, 8]:
        return 2
    elif source in [4]:
        return 3
    elif source in [5]:
        return 4
    elif source in [9, 10, 11, 16]:
        return 5
    return 6

def indigenous_to_nominal(indigenous: int) -> int:
    INDIGENOUS_OR_TSI = 4
    return 1 if indigenous == INDIGENOUS_OR_TSI else 0

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

df = pd.read_csv(
    'TestED.csv',
    dtype=object,
    encoding='unicode_escape')
df = df[SELECTED_COLUMNS]
print(f"Selected columns: {', '.join(SELECTED_COLUMNS)}")

# Ensure there are no NULL values in the data
df = df.dropna()
assert not df.isnull().values.any()

# Coerce all columns to numeric
df = df.apply(pd.to_numeric)

# Transform variables into buckets
df['age_recode'] = df['age_recode'].apply(age_to_nominal)
df['sex'] = df['sex'].apply(sex_to_nominal)
df['preferred_language_ascl'] = df['preferred_language_ascl'].apply(
    language_ascl_to_nominal)
df['ed_source_of_referral'] = df['ed_source_of_referral'].apply(
    source_of_referral_to_nominal)
df['indigenous'] = df['indigenous'].apply(indigenous_to_nominal)
df.rename(columns={
    'age_recode': 'age',
    'preferred_language_ascl': 'prefers_english',
    'dx_subcode_final': 'diagnosis',
}, inplace=True)

START_FEATURES = [
    'triage_category', 'ambulance', 'age', 'flgadmit30', 'business_hours',
    'diagnosis', 'admitted']

df.to_csv(SPECIFIC_OUTPUT_DIR + 'no-feature-selection.csv', index=False)
df[START_FEATURES].to_csv(SPECIFIC_OUTPUT_DIR +
                          'start-features.csv', index=False)

with open(SPECIFIC_OUTPUT_DIR + 'summary.txt', 'w') as f:
    with redirect_stdout(f):
        output_analytics(df)

# Derive broad diagnosis codes
df['diagnosis'] = df['diagnosis'].apply(int)
df.to_csv(BROAD_OUTPUT_DIR + 'no-feature-selection.csv', index=False)
df[START_FEATURES].to_csv(BROAD_OUTPUT_DIR + 'start-features.csv', index=False)

with open(BROAD_OUTPUT_DIR + 'summary.txt', 'w') as f:
    with redirect_stdout(f):
        output_analytics(df)