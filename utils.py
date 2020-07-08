from scipy.sparse import csr_matrix
from sklearn import preprocessing


def to_csr(df):
    """ids should be label encoded ints"""
    row_count = df.user_id.max() + 1
    col_count = df.item_id.max() + 1

    return csr_matrix(
        (df.rating, (df.user_id, df.item_id)), shape=(row_count, col_count),
    )


def encode(df):
    ue = preprocessing.LabelEncoder()
    ie = preprocessing.LabelEncoder()
    df['user_id'] = ue.fit_transform(df['user_id'])
    df['item_id'] = ie.fit_transform(df['item_id'])
    return df, ue, ie
