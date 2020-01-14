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

liveinfo_df_t = liveinfo_df_tmp.drop_duplicates(subset = "live_id", keep = "last")
userinfo_df_t = userinfo_df_tmp.drop_duplicates(subset = "uid", keep = "last")

#normalize the two dataframes
no_normalizeCol_userinfo = ['uid', 'rversion', 'deviceid', 'yyuid', 'countrycode', 'registertime', 'updatetime']
normalizeCol_userinfo = [x for x in list(userinfo_df_t) if x not in no_normalizeCol_userinfo]
no_normalizeCol_liveinfo = ['live_id', 'starttime_dt', 'starttimestamp', 'uid']
normalizeCol_liveinfo = [x for x in list(liveinfo_df_t) if x not in no_normalizeCol_liveinfo]

min_max_scaler = MinMaxScaler()
x_userinfo = userinfo_df_t[normalizeCol_userinfo].values
x_userinfo_scaled = min_max_scaler.fit_transform(x_userinfo)
x_liveinfo = liveinfo_df_t[normalizeCol_liveinfo].values
x_liveinfo_scaled = min_max_scaler.fit_transform(x_liveinfo)

userinfo_df = pd.DataFrame(x_userinfo_scaled, columns = normalizeCol_userinfo, index = userinfo_df_t.index)
userinfo_df['uid'] = userinfo_df_t['uid']
userinfo_df['rversion'] = userinfo_df_t['rversion']
userinfo_df['deviceid'] = userinfo_df_t['deviceid']
userinfo_df['yyuid'] = userinfo_df_t['yyuid']
userinfo_df['countrycode'] = userinfo_df_t['countrycode']
userinfo_df['registertime'] = userinfo_df_t['registertime']
userinfo_df['updatetime'] = userinfo_df_t['updatetime']


liveinfo_df = pd.DataFrame(x_liveinfo_scaled, columns = normalizeCol_liveinfo, index = liveinfo_df_t.index)
liveinfo_df['live_id'] = liveinfo_df_t['live_id']
liveinfo_df['starttime_dt'] = liveinfo_df_t['starttime_dt']
liveinfo_df['starttimestamp'] = liveinfo_df_t['starttimestamp']
liveinfo_df['uid'] = liveinfo_df_t['uid']

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
    feature_matrix_1.to_csv("feature_matrix_1.csv")
