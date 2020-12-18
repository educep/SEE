from typing import List
from collections import Counter, defaultdict
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

def raw_majority_vote(labels: list) -> str:
    votes = Counter(labels)
    winner, _ = votes.most_common(1)[0]
    return winner

def majority_vote(labels: list) -> str:
    """Assumes that labels are ordered from nearest to farthest."""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


if __name__ == '__main__':

    votes = Counter(['a', 'b', 'c', 'b'])
    winner, nbvotes = votes.most_common(1)[0]
    raw_majority_vote(['a', 'b', 'c', 'b'])

    data = requests.get(r'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
    txt = data.text.split('\n')
    df = pd.DataFrame(columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'],
                      index=range(len(txt)))

    for _, tt in enumerate(txt):
        crow = tt.split(',')
        if len(crow) > 1:
            df.iloc[_, :] = tt.split(',')

    df.dropna(how='all', axis=0, inplace=True)

    for _ in df.columns:
        df[_] = pd.to_numeric(df[_], errors='ignore')


    sns.pairplot(df, hue='class')
    plt.savefig('Iris_scatterplot.png')

    k = 5
    nb_train = int(0.7 * df.shape[0])
    idx = list(df.index.copy())
    random.shuffle(idx)          # because shuffle modifies the list.
    train, test = df.loc[idx[0:nb_train]], df.loc[idx[nb_train:]]

    assert train.shape[0] + test.shape[0] == df.shape[0]

    res = []
    for i in range(test.shape[0]):
        new_pt = test.iloc[i]
        distances = np.sqrt((((train.values[:, :-1] - new_pt.values[None, :-1]) ** 2)).sum(axis=1).astype(float))
        distances = pd.DataFrame(np.c_[distances, train.iloc[:, -1]], columns=['distance', 'class'])
        distances.sort_values(by='distance', inplace=True)
        res.append(majority_vote(distances.iloc[0:k, 1].values))

    final_result = test.copy()
    final_result['prediction'] = res

    """
    confusion matrix
    """
    from typing import Tuple, Dict
    confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
    num_correct = 0

    for _, row in final_result.iterrows():
        predicted = row['prediction']
        actual = row['class']
        if predicted == actual:
            num_correct += 1

        confusion_matrix[predicted, actual] += 1

    pct_correct = num_correct / final_result.shape[0]
    print(pct_correct)
    for _, val in confusion_matrix.items(): print(_, val)


    sns.pairplot(final_result, hue='class')
    sns.pairplot(final_result, hue='prediction')

    # TODO do the same thing with scikit-learn

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    neigh = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
    neigh.fit(train.iloc[:, :-1].values, train.iloc[:, -1].values)
    res_ = neigh.predict(test.iloc[:, :-1])

    confusion_matrix(y_true=test.iloc[:, -1].values, y_pred=res_)