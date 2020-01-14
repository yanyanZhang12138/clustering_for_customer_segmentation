#%% 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import featuretools as ft 
from datetime import datetime, timedelta
import pandas_profiling
from featuretools.primitives import make_agg_primitive, make_trans_primitive
from featuretools.variable_types import Text, Numeric
from featuretools.autonormalize import autonormalize as an
# %%

liveinfo_df_tmp = pd.read_csv('liveinfo_df1.csv')
userinfo_df_tmp = pd.read_csv('userinfo_df1.csv')

liveinfo_df = liveinfo_df_tmp.drop_duplicates(subset = "live_id", keep = "last")
userinfo_df = userinfo_df_tmp.drop_duplicates(subset = "uid", keep = "last")


# featuretools-generating new features
if __name__ == "__main__":
#%%
    es = ft.EntitySet(id = "user_data") #initialize an entityset by giving a id
    es = es.entity_from_dataframe(entity_id = "liveroom_info",
                            dataframe = liveinfo_df,
                            index = "live_id",
                            time_index = "starttime_dt",
                            variable_types = {"uid":ft.variable_types.Categorical}) #load the liveinfo_df dataframe as an entity
    es = es.entity_from_dataframe(entity_id = "users_info",
                            dataframe = userinfo_df,
                            index = "uid") #adding userinfo_df dataframe as another entity
    new_relationship = ft.Relationship(es["users_info"]["uid"],
                                es["liveroom_info"]["uid"]) #define the relationship between these two entitysets:(parent_entity, parent_variable, child_entity, child_variable)
    es = es.add_relationship(new_relationship)

    feature_matrix_1, feature_defs_1 = ft.dfs(entityset = es,
                                    target_entity = "users_info",
                                    trans_primitives = ["time_since_previous"],
                                    agg_primitives = ["mean", "max", "min", "std", "skew", "count", "sum"],
                                    n_jobs = 20
                                    ) #use this entityset with any functionality within featuretools


# %%
    feature_matrix_1.to_csv("feature_matrix_2.csv")
