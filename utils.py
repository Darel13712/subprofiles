from scipy.sparse import csc_matrix
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors


def to_csc(df):
    """ids should be label encoded ints"""
    row_count = df.user_id.max() + 1
    col_count = df.item_id.max() + 1

    return csc_matrix(
        (df.rating, (df.user_id, df.item_id)), shape=(row_count, col_count),
    )


def encode(df):
    ue = preprocessing.LabelEncoder()
    ie = preprocessing.LabelEncoder()
    df['user_id'] = ue.fit_transform(df['user_id'])
    df['item_id'] = ie.fit_transform(df['item_id'])
    return df, ue, ie


def get_knn(ui, k, metric='cosine', inverse=False):
    iu = ui.T
    nbrs = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1).fit(iu)
    knn = nbrs.kneighbors(iu, return_distance=False)
    knn = {i: set(items).union({i}) for i, items in enumerate(knn)}
    if inverse:
        knn = {
            base_item: {item for item in knn if base_item in knn[item]}.union(
                {base_item}
            )
            for base_item in knn
        }
    return knn
