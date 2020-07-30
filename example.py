from rs_datasets import MovieLens

from subprofiles import get_subprofiles
from utils import to_csc, encode, get_knn

df = MovieLens('20m').ratings
df, ue, ie = encode(df)
m = to_csc(df)
knn = get_knn(m, 10, 'cosine')
sp = get_subprofiles(m, knn)
