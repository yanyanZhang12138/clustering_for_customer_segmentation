# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import pairwise_distances, silhouette_samples, silhouette_score, davies_bouldin_score
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from skfeature.function.sparse_learning_based import MCFS, NDFS
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import inspect
import pickle

# %%
# data preprocessing
# load data
df_test = pd.read_csv('feature_matrix_1.csv')
df_test.fillna(df_test.mean(), inplace=True)

# normalize some columns
mean_starttime_diff_tmp = ['mean_starttime_diff']
max_starttime_diff_tmp = ['max_starttime_diff']
min_starttime_diff_tmp = ['min_starttime_diff']
std_starttime_diff_tmp = ['std_starttime_diff']
sum_starttime_diff_tmp = ['sum_starttime_diff']
min_max_scaler = MinMaxScaler()
norm_mean_starttime_diff = df_test[mean_starttime_diff_tmp].values
norm_max_starttime_diff = df_test[max_starttime_diff_tmp].values
norm_min_starttime_diff = df_test[min_starttime_diff_tmp].values
norm_std_starttime_diff = df_test[std_starttime_diff_tmp].values
norm_sum_starttime_diff = df_test[sum_starttime_diff_tmp].values

mean_starttime_diff_scaled = min_max_scaler.fit_transform(
    norm_mean_starttime_diff)
max_starttime_diff_scaled = min_max_scaler.fit_transform(
    norm_max_starttime_diff)
min_starttime_diff_scaled = min_max_scaler.fit_transform(
    norm_min_starttime_diff)
std_starttime_diff_scaled = min_max_scaler.fit_transform(
    norm_std_starttime_diff)
sum_starttime_diff_scaled = min_max_scaler.fit_transform(
    norm_sum_starttime_diff)

df_test['mean_starttime_diff'] = mean_starttime_diff_scaled
df_test['max_starttime_diff'] = max_starttime_diff_scaled
df_test['min_starttime_diff'] = min_starttime_diff_scaled
df_test['std_starttime_diff'] = std_starttime_diff_scaled
df_test['sum_starttime_diff'] = sum_starttime_diff_scaled

X_test_full = df_test[['activedays', 'followers', 'following', 'recent30days_login_cnt', 'user_level', 'avg_bantime_min09', 'avg_people09', 'avg_heart09', 'avg_gift09', 'avg_barrage09', 'avg_msg09', 'avg_totaltime09', 'avg_bantime_min0609',
                       'avg_people0609', 'avg_heart0609', 'avg_gift0609', 'avg_barrage0609', 'avg_msg0609', 'avg_totaltime0609', 'punish_live_ban0609', 'punish_ignore0609', 'avg_bantime_min0406', 'avg_people0406',
                       'avg_heart0406', 'avg_gift0406', 'avg_barrage0406', 'avg_msg0406', 'avg_totaltime0406', 'watch_punish_a09', 'watch_punish_warn09', 'watch_punish_special_a09', 'watch_punish_a0609', 'watch_punish_warn0609',
                       'watch_punish_special_a0406', 'rejection_cnt09', 'rejection_cnt0609', 'rejection_cnt0406', 'im_frequency09', 'im_frequency0609', 'im_frequency0406', 'keyword_rejection_frequency09', 'keyword_rejection_frequency0609',
                       'keyword_rejection_frequency0406', 'forum_post_cnt09', 'forum_post_cnt0609', 'forum_post_cnt0406', 'review_frequency', 'punish_b', 'punish_warn', 'punish_live_ban', 'punish_firstreview10min', 'punish_special_a', 'status_b',
                       'status_warn', 'status_special_a', 'tap_porn', 'watch_punish_a', 'watch_punish_b', 'watch_punish_warn', 'watch_punish_special_a', 'action_screen', 'rejection_cnt', 'im_frequency', 'keyword_rejection_frequency',
                       'forum_post_cnt', 'forum_comment_cnt', 'bar_banned_frequency', 'bar_deleted_frequency', 'mean_normal_gift', 'mean_follow_owner_room', 'mean_msg_cnt', 'mean_heartcnt', 'mean_barragecnt', 'mean_giftcnt', 'mean_totaltime',
                       'max_msgcnt', 'max_heartcnt', 'max_barragecnt', 'max_giftcnt', 'max_totaltime', 'max_totaltime', 'min_msgcnt', 'min_heartcnt', 'min_giftcnt', 'min_starttimestamp', 'std_msgcnt',
                       'std_heartcnt', 'std_barragecnt', 'std_giftcnt', 'std_totaltime', 'skew_msgcnt', 'skew_heartcnt', 'skew_barragecnt', 'skew_giftcnt', 'skew_totaltime', 'live_cnt', 'sum_normal_gift', 'sum_follow_owner_room', 'sum_msgcnt',
                       'sum_heartcnt', 'sum_barragecnt', 'sum_giftcnt', 'sum_totaltime', 'mean_starttime_diff', 'max_starttime_diff', 'min_starttime_diff', 'std_starttime_diff', 'skew_starttime_diff', 'sum_starttime_diff']]

X_test_full = X_test_full.astype(float)
X_test_full = df_test[['activedays', 'followers', 'following', 'recent30days_login_cnt', 'user_level', 'avg_bantime_min09', 'avg_people09', 'avg_heart09', 'avg_gift09', 'avg_barrage09', 'avg_msg09', 'avg_totaltime09', 'avg_bantime_min0609',
                       'avg_people0609', 'avg_heart0609', 'avg_gift0609', 'avg_barrage0609', 'avg_msg0609', 'avg_totaltime0609', 'punish_live_ban0609', 'punish_ignore0609', 'avg_bantime_min0406', 'avg_people0406',
                       'avg_heart0406', 'avg_gift0406', 'avg_barrage0406', 'avg_msg0406', 'avg_totaltime0406', 'watch_punish_a09', 'watch_punish_warn09', 'watch_punish_special_a09', 'watch_punish_a0609', 'watch_punish_warn0609',
                       'watch_punish_special_a0406', 'rejection_cnt09', 'rejection_cnt0609', 'rejection_cnt0406', 'im_frequency09', 'im_frequency0609', 'im_frequency0406', 'keyword_rejection_frequency09', 'keyword_rejection_frequency0609',
                       'keyword_rejection_frequency0406', 'forum_post_cnt09', 'forum_post_cnt0609', 'forum_post_cnt0406', 'review_frequency', 'punish_b', 'punish_warn', 'punish_live_ban', 'punish_firstreview10min', 'punish_special_a', 'status_b',
                       'status_warn', 'status_special_a', 'tap_porn', 'watch_punish_a', 'watch_punish_b', 'watch_punish_warn', 'watch_punish_special_a', 'action_screen', 'rejection_cnt', 'im_frequency', 'keyword_rejection_frequency',
                       'forum_post_cnt', 'forum_comment_cnt', 'bar_banned_frequency', 'bar_deleted_frequency', 'mean_normal_gift', 'mean_follow_owner_room', 'mean_msg_cnt', 'mean_heartcnt', 'mean_barragecnt', 'mean_giftcnt', 'mean_totaltime',
                       'max_msgcnt', 'max_heartcnt', 'max_barragecnt', 'max_giftcnt', 'max_totaltime', 'max_totaltime', 'min_msgcnt', 'min_heartcnt', 'min_giftcnt', 'min_starttimestamp', 'std_msgcnt',
                       'std_heartcnt', 'std_barragecnt', 'std_giftcnt', 'std_totaltime', 'skew_msgcnt', 'skew_heartcnt', 'skew_barragecnt', 'skew_giftcnt', 'skew_totaltime', 'live_cnt', 'sum_normal_gift', 'sum_follow_owner_room', 'sum_msgcnt',
                       'sum_heartcnt', 'sum_barragecnt', 'sum_giftcnt', 'sum_totaltime', 'mean_starttime_diff', 'max_starttime_diff', 'min_starttime_diff', 'std_starttime_diff', 'skew_starttime_diff', 'sum_starttime_diff']].applymap(lambda x: '%.4f' % x)

X_test_full = X_test_full.astype(float)

# shuffle the dataset and split the dataset into 10 subsamples with replacement


def rand_sample(dataset, ratio='num'):
    X_list = list(range(10))
    for i in X_list:
        X_test_shuffle = shuffle(X_test_full)
        X_list[i] = X_test_shuffle.sample(frac=ratio).to_numpy()
    return X_list


X_testset = rand_sample(X_test_full, ratio=0.1)

# dynamic naming generation


def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]


prepare_list = locals()

# generate the ranked varialbe list


def get_variable_rank(array1, array2):
    lap_ranked_var = {'variable': []}
    for obj in array1:
        lap_ranked_var['variable'].append(array2[obj])
    return lap_ranked_var

# create the tables for silhouette, Cal, Davies seperately


def generate_criteria_tb(dict_name='dict_name', col_name='col_name'):
    criteria_tb = pd.DataFrame()
    tmp_list  = range(0, 10, 1)
    for i in tmp_list:
        criteria_tb[col_name + str(i)] = prepare_list[dict_name +
                                                 str(i)][col_name]
    return criteria_tb

# feature selection methodology: lap score ranking


def lapscore_main():

    # iterate the whole process for 10 times
    for index, subsample in enumerate(X_testset):

        # construct affinity matrix
        kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn",
                    "weight_mode": "heat_kernel", "k": 5, 't': 1}
        W = construct_W.construct_W(subsample, **kwargs_W)

        # obtain the scores of  features
        idx = lap_score.lap_score(subsample, mode="rank", W=W)
        # obtian the array of variables through ranking
        X_col_list = X_test_full.columns.values.tolist()
        prepare_list['lap_ranked_Xtestset' +
                     str(index)] = get_variable_rank(idx, X_col_list)
        ranked_var_filename = 'lap_ranked_Xtestset' + str(index) + '.txt'
        f_rank = open(ranked_var_filename, 'w')
        f_rank.write(str(prepare_list['lap_ranked_Xtestset' + str(index)]))
        f_rank.close()

        # perform evaluation on clustering task
        range_num_fea = range(10, 210, 10)  # number of selected features
        range_n_clusters = [3, 4, 5, 6, 7, 8, 9, 10]  # number of clusters

        # dynamic generating dictionaries to store results
        prepare_list['lapscore_criteria' +
                     str(index)] = {'silhouette_score': [], 'ch_score': [], 'db_score': []}

        # deciding optimal value for num_cluster and the optimal number of selected features

        for n_cluster in range_n_clusters:

            for num_features in range_num_fea:
                # obtain the dataset on the selected features
                selected_features = subsample[:, idx[0:num_features]]

                # initialize the clusterer with n_clusters value and a random generator
                # seed of 10 for reproducbility
                clusterer = KMeans(
                    n_clusters=n_cluster, random_state=10)
                cluster_labels = clusterer.fit_predict(selected_features)

                # the silhouette_score gives the average value for all the samples
                # this gives a perspective into the density and separation of the formed clusters
                silhouette_avg = metrics.silhouette_score(
                    selected_features, cluster_labels, metric='euclidean')
                # write the content into the dict
                prepare_list['lapscore_criteria' +
                             str(index)]['silhouette_score'].append(silhouette_avg)
                # in normal usage, the Calinski-Harabasz index is applied to the results of a cluster analysis
                ch_idx = metrics.calinski_harabasz_score(
                    selected_features, cluster_labels)
                # write the content into the dict
                prepare_list['lapscore_criteria' + str(index)
                             ]['ch_score'].append(ch_idx)
                # in normal usage, the Davies-Bouldin index is applied to the results of a cluster analysis
                db_idx = davies_bouldin_score(
                    selected_features, cluster_labels)
                # write the content into the dict
                prepare_list['lapscore_criteria' +
                             str(index)]['db_score'].append(db_idx)

                print("subset No.", index, ","
                      "For n_clusters =", n_cluster, ","
                      "For num_features =", num_features, ","
                      "the average silhouette_score is: ", silhouette_avg, ","
                      "the Calinski-Harabasz index is: ", ch_idx, ","
                      "the Davies-Bouldin index is: ", db_idx)

    lapscore_silhouette_score = generate_criteria_tb(
        dict_name='lapscore_criteria', col_name='silhouette_score')
    lapscore_Calinski_Harabasz_index = generate_criteria_tb(
        dict_name='lapscore_criteria', col_name='ch_score')
    lapscore_Davies_Bouldin_index = generate_criteria_tb(
        dict_name='lapscore_criteria', col_name='db_score')

    lapscore_silhouette_score.to_csv(
        'lapscore_silhouette_score.csv', index=False)
    lapscore_Calinski_Harabasz_index.to_csv(
        'lapscore_Calinski_Harabasz_index.csv', index=False)
    lapscore_Davies_Bouldin_index.to_csv(
        'lapscore_Davies_Bouldin_index.csv', index=False)


def generation_kmeans_model(dataframe, n_clusters = 'num_cluster'):
    df_list = dataframe.to_numpy()
    kmeans_model = KMeans(init = 'k-means++', n_clusters = n_clusters, random_state = 10).fit(df_list)
    pred_labels = kmeans_model.labels_
    return kmeans_model, pred_labels


if __name__ == '__main__':

    #lapscore_main()
    kmeans_model, pred_labels = generation_kmeans_model(X_test_full, n_clusters = 3)
    filename = 'kmeans_model.sav'
    pickle.dump(kmeans_model, open(filename, 'wb'))
    np.savetxt('pred_labels.csv', pred_labels, delimiter = ',')


