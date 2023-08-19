import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')

DATA_SOURCE = "/home/bellaando/thesis23/TestED.csv"

print('hello! :)')
df = pd.read_csv(DATA_SOURCE, encoding='unicode_escape')
print('data read in!')
for col in df.columns:

    # print({col}, 'is being worked on.')
    if col == 'repres7days' or ((col.startswith('PROCEDURE') or col.startswith('DIAGNOSIS')) and (col.endswith('P') == False)):
        continue

    if col != 'PREFERRED_LANGUAGE_ASCL':
        continue

    # Cross-tabulate each attribute against class attribute
    tab = pd.crosstab(df[col], df['repres7days'], dropna=False)

    # if len(tab.index) > 100:
    #     continue

    try:
        ax = tab.plot(kind='bar', stacked=True)
        ax.set_ylabel("Num instances")
    except:
        print({col}, 'could not be plotted.')
        continue

    print('figure saving...')
    plt.savefig(
        f'/home/bellaando/thesis23/distribution-plots/{col}.png',
        bbox_inches='tight')
    plt.close()
    print('saved fig:', {col})