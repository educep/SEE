import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
query(), assign(), pivot_table(), pipe(), melt()
"""

df = pd.DataFrame({'A': {0: 'a', 1: 'b', 2: 'c'},
                   'B': {0: 1, 1: 3, 2: 5},
                   'C': {0: 2, 1: 4, 2: 6}})

pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])

df.assign(new=1, new2=lambda df: df['B']**2)

db = pd.DataFrame(np.round(100*np.random.random((100,3)), 0), columns=['age1', 'age2', 'age3'])
bins = [0, 13, 19, 61, 100]
labels = ['<12', 'Teen', 'Adult', 'Older']

db = db.assign(ageGroup=lambda db: pd.cut(db['age1'], bins=bins, labels=labels, include_lowest=True))