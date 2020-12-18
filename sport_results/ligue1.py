from bs4 import BeautifulSoup
import requests as req
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns


results = dict()
for year in range(1934,2020):
    resp = req.get(f'http://www.footballstats.fr/resultat-ligue1-{str(year)}.html')
    bs = BeautifulSoup(resp.text, 'html.parser')
    table = bs.find(lambda tag: tag.name=='table')
    rows = table.findAll(lambda tag: tag.name=='tr')

    all_ = pd.DataFrame()
    try:
        for kk in range(1, len(rows)):
            ri = [_.replace(' ', '') for _ in rows[kk].get_text().split('\n')]
            all_ = pd.concat([all_, pd.DataFrame(ri[3:-1], columns=[ri[2]])], axis=1)

        all_.index = all_.columns
        all_ = all_.T
        np.fill_diagonal(all_.values, np.nan)
        results[year] = all_
    except Exception as e:
        print(year, str(e))



to_store = pd.DataFrame()
for _ in results.keys():
    try:
        aa = results[_].stack(level=0).reset_index(drop=False)
        aa.columns = ['Home', 'Visit', 'Score']
        aa['Year'] = _
        to_store = pd.concat([to_store, aa], axis=0)
    except Exception as e:
        print(_, 'bad things happened\n', str(e))

# le local est en ligne, visiteur en colonne
to_store.to_pickle('historical_results_ligue1.pkl')
to_store.loc[to_store.Score == '-']

scores = Counter(to_store['Score'])
# scores = {k: v for k, v in sorted(scores.items(), key=lambda x: x[1])}
# scores = pd.DataFrame(list(scores.values()), index=scores.keys(), columns=['freq'])
# scores.sort_values(by='freq', inplace=True)

stats = to_store.loc[to_store.Score != "-"].groupby(['Year', 'Score'])['Score'].agg('count').to_frame(name='Count')
stats.reset_index(drop=False, inplace=True)
# stats.loc[stats.Score == '-']
nb_matches = to_store.groupby(['Year'])['Score'].count().reset_index()
nb_matches.columns = ['Year', 'NbMatch']

stats = stats.merge(nb_matches, how='left', on=['Year'])
stats['Freq'] = 100 * stats['Count'] / stats['NbMatch']

all_avg = stats.groupby(['Score'])['Freq'].mean()
last_avg = stats.loc[stats.Year >= 2000].groupby(['Score'])['Freq'].mean()

averages = pd.concat({'Full_Period': all_avg, 'From_2000': last_avg}, axis=1).fillna(0.)
averages.sort_values(by=['Full_Period'], inplace=True, ascending=False)
averages.iloc[0:21].plot.bar()

# by team
teams = ['GUINGAMP', 'LYON', 'MARSEILLE', 'PARIS-SG', 'MONACO']
stats = to_store.loc[to_store.Score != "-"].groupby(['Year', 'Home', 'Score'])['Score'].agg('count').to_frame(name='Count')
stats.reset_index(drop=False, inplace=True)
nb_matches = to_store.groupby(['Year', 'Home'])['Score'].count().reset_index()
nb_matches.columns = ['Year', 'Home', 'NbMatch']
stats = stats.merge(nb_matches, how='left', on=['Year', 'Home'])
stats['Freq'] = 100 * stats['Count'] / stats['NbMatch']
stats = stats.loc[(stats.Home.isin(teams)) & (stats.Year >= 2000)]
bb = stats.set_index(['Year', 'Home', 'Score'])['Freq'].unstack(['Home', 'Score']).fillna(0.).mean().reset_index()
bb.columns = ['Home', 'Score', 'Freq']
scores_ = bb.groupby(['Score'])['Freq'].mean().sort_values(ascending=False).iloc[0:20].index
bb = bb.sort_values(by='Freq', ascending=False)
sns.barplot(x="Score", y='Freq',  hue="Home", data=bb.loc[bb.Score.isin(scores_)])