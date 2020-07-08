from rs_datasets import MovieLens

from subprofiles import subprofiles
from utils import to_csr, encode

df = MovieLens().ratings
df, ue, ie = encode(df)
m = to_csr(df)
sp = subprofiles(m)
