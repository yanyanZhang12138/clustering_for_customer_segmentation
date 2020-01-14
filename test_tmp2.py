#%%
import pandas as pd 
import numpy as np 

userinfo_df_tmp = pd.read_csv('userinfo_df1.csv')
userinfo_df_t = userinfo_df_tmp.drop_duplicates(subset = "uid", keep = "last")

userinfo_df_t.to_csv('userinfo_updated.csv')




# %%
