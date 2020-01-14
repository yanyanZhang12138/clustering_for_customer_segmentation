import pandas as pd 
import numpy as np 

userinfo_df = pd.read_csv('userinfo_df1.csv')
df_label = pd.read_csv('feature_matrix_label.csv')

userinfo_df = userinfo_df.drop_duplicates(subset = 'uid', keep = 'last')
df_label = df_label.drop_duplicates(subset = 'uid', keep = 'last')
userinfo_df['uid'] = userinfo_df['uid'].astype(str)
df_label['uid'] = df_label['uid'].astype(str)

print(userinfo_df.count)
print(df_label.count)

df_label_tmp = df_label[['uid', 'mean_starttime_diff', 'min_starttime_diff', 'max_starttime_diff', 'std_starttime_diff', 'sum_starttime_diff', 'pred_label']]
userinfo_df_cols = userinfo_df.columns.values.tolist()

print(df_label_tmp)
finalized_df = pd.merge(userinfo_df, df_label_tmp, how = 'inner', left_on = userinfo_df_cols, right_on = ['uid', 'mean_starttime_df', 'min_starttime_df', 'max_starttime_df', 'std_starttime_df', 'sum_starttime_df', 'pred_label'])
print(finalized_df.head())

#finalized_df.to_csv('finalized_kmeans_df.csv')


