import os
import pathlib
from pandas import DataFrame

from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)
wine_quality_df = wine_quality.data.original

wine_quality_dirpath = pathlib.Path('./Wine_Quality')
os.makedirs(wine_quality_dirpath, exist_ok=True)
wine_quality_path = wine_quality_dirpath / 'data.csv'
wine_quality_df.to_csv(wine_quality_path)

abalone_data = fetch_ucirepo(id=1)
abalone_df = abalone_data.data.original

abalone_dirpath = pathlib.Path('./Abalone')
os.makedirs(abalone_dirpath, exist_ok=True)
abalone_path = abalone_dirpath / 'data.csv'
abalone_df.to_csv(abalone_path)

breast_cancer_diagnostic_wiscon = fetch_ucirepo(id=17)
wiscon_df = breast_cancer_diagnostic_wiscon.data.original

wiscon_dirpath = pathlib.Path('./WI_Cancer_Diag')
os.makedirs(wiscon_dirpath, exist_ok=True)
wiscon_path = wiscon_dirpath / 'data.csv'
wiscon_df.to_csv(wiscon_path)