#%% 
import pandas as pd 
import numpy as np
#import featuretools as ft 
from datetime import datetime, timedelta
import pandas_profiling
#from featuretools.primitive import make_agg_primitive, make_trans_primitive
#from featuretools.variable_types import Text, Numeric
# %%
liveinfo_df = pd.read_csv('liveinfo_highquality0929.txt', delim_whitespace = True, header = None)
userinfo_df = pd.read_csv('userinfo_complete_test1.csv', header = None, names = ['rversion', 'uid', 'deviceid', 'status', 'registertime', 
'updatetime', 'user_type', 'yyuid', 'activedays', 'followers', 'following', 'countrycode', 
'recent30days_login_cnt', 'user_level', 'review_frequency09', 'avg_bantime_min09', 'avg_people09', 
'avg_heart09', 'avg_gift09', 'avg_barrage09', 'avg_msg09', 'avg_totaltime09', 'type_1_09', 
'type_2_09', 'type_3_09', 'punish_0_09', 'punish_a09', 'punish_b09', 'punish_warn09', 
'punish_live_ban09', 'punish_block09', 'punish_ignore09', 'punish_firstreview10min09', 
'punish_account_ban09', 'punish_actionscreen_ban09', 'punish_special_a09', 'punish_child09', 
'source_alarm09', 'source_customized_ban09', 'source_inspect09', 'source_first09', 'source_final09',
'source_user_ban09', 'source_block_manage09', 'source_parallel09', 'status_ignore09', 'status_a09', 'status_b09',
'status_warn09', 'status_ban09', 'status_special_a09', 'status_child09', 'tap_porn09', 'tap_violencehorror09', 'tap_alcoholgamble09',
'tap_politicsreligion09','tap_meaningless09', 'tap_infringement09', 'review_frequency0609','avg_bantime_min0609', 'avg_people0609', 
'avg_heart0609', 'avg_gift0609', 'avg_barrage0609', 'avg_msg0609', 'avg_totaltime0609', 'type_1_0609', 'type_2_0609', 'type_3_0609',
'punish_0_0609', 'punish_a0609', 'punish_b0609', 'punish_warn0609', 'punish_live_ban0609', 'punish_block0609',
'punish_ignore0609', 'punish_firstreview10min0609', 'punish_account_ban0609', 'punish_actionscreen_ban0609',
'punish_special_a0609', 'punish_child0609', 'source_alarm0609', 'source_customized_ban0609', 'source_inspect0609',
'source_first0609', 'source_final0609', 'source_user_ban0609', 'source_block_manage0609', 'source_parallel0609', 'status_ignore0609',
'status_a0609', 'status_b0609', 'status_warn0609', 'status_ban0609', 'status_special_a0609', 'status_child0609', 
'tap_porn0609', 'tap_violencehorror0609', 'tap_alcoholgamble0609', 'tap_politicsreligion0609', 'tap_meaningless0609',
'tap_infringement0609', 'review_frequency0406', 'avg_bantime_min0406', 'avg_people0406', 'avg_heart0406', 'avg_gift0406',
'avg_barrage0406', 'avg_msg0406', 'avg_totaltime0406', 'type_1_0406', 'type_2_0406', 'type_3_0406',
'punish_0_0406', 'punish_a0406', 'punish_b0406', 'punish_warn0406', 'punish_live_ban0406', 'punish_block0406',
'punish_ignore0406', 'punish_firstreview10min0406', 'punish_account_ban0406', 'punish_actionscreen_ban0406',
'punish_special_a0406', 'punish_child0406', 'source_alarm0406', 'source_customized_ban0406', 'source_inspect0406',
'source_first0406', 'source_final0406', 'source_user_ban0406', 'source_block_manage0406', 'source_parallel0406',
'status_ignore0406', 'status_a0406', 'status_b0406', 'status_warn0406', 'status_ban0406', 'status_special_a0406',
'status_child0406', 'tap_porn0406', 'tap_violencehorror0406', 'tap_alcoholgamble0406', 'tap_politicsreligion0406',
'tap_meaningless0406', 'tap_infringement0406', 'watch_punish_a09', 'watch_punish_b09', 'watch_punish_warn09',
'watch_punish_special_a09', 'watch_punish_a0609', 'watch_punish_b0609', 'watch_punish_warn0609', 
'watch_punish_special_a0609', 'watch_punish_a0406', 'watch_punish_b0406', 'watch_punish_warn0406',
'watch_punish_special_a0406', 'action_screen09','action_screen0609', 'action_screen0406', 'avg_ticket_a09',
'avg_ticket_b09', 'avg_ticket_warn09', 'avg_ticket_child09', 'avg_ticket_special_a09', 'avg_ticket_ignore09',
'avg_ticket_a0609', 'avg_ticket_b0609', 'avg_ticket_warn0609', 'avg_ticket_child0609', 'avg_ticket_special_a0609',
'avg_ticket_ignore0609', 'avg_ticket_a0406', 'avg_ticket_b0406', 'avg_ticket_warn0406', 'avg_ticket_child0406',
'avg_ticket_special_a0406', 'avg_ticket_ignore0406', 'avg_vm_a09', 'avg_vm_b09', 'avg_vm_warn09', 'avg_vm_child09',
'avg_vm_special_a09', 'avg_vm_ignore09', 'avg_vm_a0609', 'avg_vm_b0609', 'avg_vm_warn0609', 'avg_vm_child0609',
'avg_vm_special_a0609', 'avg_vm_ignore0609', 'avg_vm_a0406', 'avg_vm_b0406', 'avg_vm_warn0406', 'avg_vm_child0406',
'avg_vm_special_a0406', 'avg_vm_ignore0406', 'rejection_cnt09', 'rejection_cnt0609', 'rejection_cnt0406', 'im_frequency09',
'im_frequency0609', 'im_frequency0406', 'keyword_rejection_frequency09', 'keyword_rejection_frequency0609',
'keyword_rejection_frequency0406', 'im_banned_frequency09', 'im_banned_frequency0609', 'im_banned_frequency0406',
'forum_rejection_frequency09', 'forum_rejection_frequency0609', 'forum_rejection_frequency0406',
'forum_post_cnt09', 'forum_post_cnt0609', 'forum_post_cnt0406', 'forum_comment_cnt09', 'forum_comment_cnt0609',
'forum_comment_cnt0406', 'bar_banned_frequency09', 'bar_banned_frequency0609', 'bar_banned_frequency0406',
'bar_deleted_frequency09', 'bar_deleted_frequency0609', 'bar_deleted_frequency0406', 'bar_ban_device_frequency09',
'bar_ban_device_frequency0609', 'bar_ban_device_frequency0406', 'bar_ban_user_frequency09', 'bar_ban_user_frequency0609',
'bar_ban_user_frequency0406'])
# %%
liveinfo_df.columns = ['uid', 'starttimestamp', 'totaltime', 'heartcount', 'giftcount', 'barragecount', 'msgcount', 'audience_cnt', 'mic_cnt', 'mic_totaltime', 'mic_stopreasonviolation', 'ticket_metric', 'normal_gift_action', 'silenced_action', 'real_user_enter_room_action', 'notify_follow_action', 'enter_room_action', 'follow_owner_room', 'share_living_action', 'light_heart_action']
print(liveinfo_df.head())
print(userinfo_df.head())
# %%
liveinfo_df= liveinfo_df.replace(r'\\N', 0, regex = True)
userinfo_df.fillna(0, inplace = True)
# %%
liveinfo_df.info()
userinfo_df.info()
# %%
liveinfo_df['uid'] = liveinfo_df['uid'].astype(str)
liveinfo_df['starttimestamp'] = liveinfo_df['starttimestamp'].astype(str)
liveinfo_df['starttime_dt'] = pd.to_datetime(liveinfo_df['starttimestamp'], unit= 's')
liveinfo_df['live_id'] = liveinfo_df['uid'] + liveinfo_df['starttimestamp']
liveinfo_df['mic_totaltime'] = liveinfo_df['mic_totaltime'].astype(int)
liveinfo_df['ticket_metric'] = liveinfo_df['ticket_metric'].astype(float)
userinfo_df['rversion'] = userinfo_df['rversion'].astype(str)
userinfo_df['uid'] = userinfo_df['uid'].astype(str)
userinfo_df['deviceid'] = userinfo_df['deviceid'].astype(str)
userinfo_df['status'] = userinfo_df['status'].astype(str)
userinfo_df['registertime'] = pd.to_datetime(userinfo_df['registertime'])
userinfo_df['updatetime'] = pd.to_datetime(userinfo_df['updatetime'])
userinfo_df['user_type'] = userinfo_df['user_type'].astype(str)
userinfo_df['yyuid'] = userinfo_df['yyuid'].astype(str)
userinfo_df['countrycode'] = userinfo_df['countrycode'].astype(str)
userinfo_df['avg_ticket_a0406'] = userinfo_df['avg_ticket_a0406'].astype(int)
userinfo_df['avg_ticket_a0609'] = userinfo_df['avg_ticket_a0609'].astype(int)
userinfo_df['avg_ticket_a09'] = userinfo_df['avg_ticket_a09'].astype(int)
userinfo_df['avg_ticket_b0609'] = userinfo_df['avg_ticket_b0609'].astype(int)
userinfo_df['avg_ticket_child0406'] = userinfo_df['avg_ticket_child0406'].astype(int)
userinfo_df['avg_ticket_child0609'] = userinfo_df['avg_ticket_child0609'].astype(int)
userinfo_df['avg_ticket_child09'] = userinfo_df['avg_ticket_child09'].astype(int)
userinfo_df['avg_ticket_special_a0406'] = userinfo_df['avg_ticket_special_a0406'].astype(int)
userinfo_df['avg_ticket_warn0609'] = userinfo_df['avg_ticket_warn0609'].astype(int)
userinfo_df['avg_ticket_warn0406'] = userinfo_df['avg_ticket_warn0406'].astype(int)
userinfo_df['avg_vm_a0406'] = userinfo_df['avg_vm_a0406'].astype(int)
userinfo_df['avg_vm_a0609'] = userinfo_df['avg_vm_a0609'].astype(int)
userinfo_df['avg_vm_a09'] = userinfo_df['avg_vm_a09'].astype(int)
userinfo_df['avg_vm_b0406'] = userinfo_df['avg_vm_b0406'].astype(int)
userinfo_df['avg_vm_child0406'] = userinfo_df['avg_vm_child0406'].astype(int)
userinfo_df['avg_vm_child0609'] = userinfo_df['avg_vm_child0609'].astype(int)
userinfo_df['avg_vm_child09'] = userinfo_df['avg_vm_child09'].astype(int)
userinfo_df['avg_vm_ignore0406'] = userinfo_df['avg_vm_ignore0406'].astype(int)
userinfo_df['avg_vm_special_a0406'] = userinfo_df['avg_vm_special_a0406'].astype(int)
userinfo_df['avg_vm_warn0406'] = userinfo_df['avg_vm_warn0406'].astype(int)
userinfo_df['bar_ban_device_frequency0406'] = userinfo_df['bar_ban_device_frequency0406'].astype(int)
userinfo_df['bar_ban_device_frequency0609'] = userinfo_df['bar_ban_device_frequency0609'].astype(int)
userinfo_df['bar_ban_device_frequency09'] = userinfo_df['bar_ban_device_frequency09'].astype(int)
userinfo_df['bar_ban_user_frequency0406'] = userinfo_df['bar_ban_user_frequency0406'].astype(int)
userinfo_df['bar_ban_user_frequency0609'] = userinfo_df['bar_ban_user_frequency0609'].astype(int)
userinfo_df['bar_ban_user_frequency09'] = userinfo_df['bar_ban_user_frequency09'].astype(int)
userinfo_df['bar_banned_frequency0406'] = userinfo_df['bar_banned_frequency0406'].astype(int)
userinfo_df['bar_deleted_frequency0406'] = userinfo_df['bar_deleted_frequency0406'].astype(int)
userinfo_df['bar_deleted_frequency0609'] = userinfo_df['bar_deleted_frequency0609'].astype(int)
userinfo_df['forum_comment_cnt09'] = userinfo_df['forum_comment_cnt09'].astype(int)
userinfo_df['forum_rejection_frequency0406'] = userinfo_df['forum_rejection_frequency0406'].astype(int)
userinfo_df['forum_rejection_frequency0609'] = userinfo_df['forum_rejection_frequency0609'].astype(int)
userinfo_df['im_banned_frequency0609'] = userinfo_df['im_banned_frequency0609'].astype(int)
userinfo_df['im_banned_frequency09'] = userinfo_df['im_banned_frequency09'].astype(int)
userinfo_df['punish_a09'] = userinfo_df['punish_a09'].astype(int)
userinfo_df['punish_child0406'] = userinfo_df['punish_child0406'].astype(int)
userinfo_df['tap_politicsreligion0406'] = userinfo_df['tap_politicsreligion0406'].astype(int)
userinfo_df['tap_politicsreligion0609'] = userinfo_df['tap_politicsreligion0609'].astype(int)
userinfo_df['tap_politicsreligion09'] = userinfo_df['tap_politicsreligion09'].astype(int)

#%%
#userinfo_dfnew = userinfo_df[['uid', 'deviceid', 'status', 'registertime', 'updatetime', 'user_type', 'yyuid', 'activedays', 'followers', 'following', 'countrycode', 'recent30days_login_cnt', 'user_level', '']].copy()
#userinfo_dfnew['']
def aggToSemiannual(df, col1, col2, col3):
    df['col_total'] = col1+col2+col3
    return (df['col_total'])

#%%
userinfo_df['review_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['review_frequency09'], userinfo_df['review_frequency0609'], userinfo_df['review_frequency0406'])
userinfo_df['punish_a'] = aggToSemiannual(userinfo_df, userinfo_df['punish_a09'], userinfo_df['punish_a0609'], userinfo_df['punish_a0406'])
userinfo_df['punish_b'] = aggToSemiannual(userinfo_df, userinfo_df['punish_b09'], userinfo_df['punish_b0609'], userinfo_df['punish_b0406'])
userinfo_df['punish_warn'] = aggToSemiannual(userinfo_df, userinfo_df['punish_warn09'], userinfo_df['punish_warn0609'], userinfo_df['punish_warn0406'])
userinfo_df['punish_live_ban'] = aggToSemiannual(userinfo_df, userinfo_df['punish_live_ban09'], userinfo_df['punish_live_ban0609'], userinfo_df['punish_live_ban0406'])
userinfo_df['punish_block'] = aggToSemiannual(userinfo_df, userinfo_df['punish_block09'], userinfo_df['punish_block0609'], userinfo_df['punish_block0406'])
userinfo_df['punish_ignore'] = aggToSemiannual(userinfo_df, userinfo_df['punish_ignore09'], userinfo_df['punish_ignore0609'], userinfo_df['punish_ignore0406'])
userinfo_df['punish_firstreview10min'] = aggToSemiannual(userinfo_df, userinfo_df['punish_firstreview10min09'], userinfo_df['punish_firstreview10min0609'], userinfo_df['punish_firstreview10min0406'])
userinfo_df['punish_account_ban'] = aggToSemiannual(userinfo_df, userinfo_df['punish_account_ban09'], userinfo_df['punish_account_ban0609'], userinfo_df['punish_account_ban0406'])
userinfo_df['punish_actionscreen_ban'] = aggToSemiannual(userinfo_df, userinfo_df['punish_actionscreen_ban09'], userinfo_df['punish_actionscreen_ban0609'], userinfo_df['punish_actionscreen_ban0406'])
userinfo_df['punish_special_a'] = aggToSemiannual(userinfo_df, userinfo_df['punish_special_a09'], userinfo_df['punish_special_a0609'], userinfo_df['punish_special_a0406'])
userinfo_df['punish_child'] = aggToSemiannual(userinfo_df, userinfo_df['punish_child09'], userinfo_df['punish_child0609'], userinfo_df['punish_child0406'])
#userinfo_df['source_alarm'] = aggToSemiannual(userinfo_df, userinfo_df['source_alarm09'], userinfo_df['source_alarm0609'], userinfo_df['source_alarm0406'])
#userinfo_df['source_customized_ban'] = aggToSemiannual(userinfo_df, userinfo_df['source_customized_ban09'], userinfo_df['source_customized_ban0609'], userinfo_df['source_customized_ban0406'])
#userinfo_df['source_inspect'] = aggToSemiannual(userinfo_df, userinfo_df['source_inspect09'], userinfo_df['source_inspect0609'], userinfo_df['source_inspect0406'])
#userinfo_df['source_first'] = aggToSemiannual(userinfo_df, userinfo_df['source_first09'], userinfo_df['source_first0609'], userinfo_df['source_first0406'])
#userinfo_df['source_final'] = aggToSemiannual(userinfo_df, userinfo_df['source_final09'], userinfo_df['source_final0609'], userinfo_df['source_final0406'])
#userinfo_df['source_user_ban'] = aggToSemiannual(userinfo_df, userinfo_df['source_user_ban09'], userinfo_df['source_user_ban0609'], userinfo_df['source_user_ban0406'])
#userinfo_df['source_block_manage'] = aggToSemiannual(userinfo_df, userinfo_df['source_block_manage09'], userinfo_df['source_block_manage0609'], userinfo_df['source_block_manage0406'])
#userinfo_df['source_parallel'] = aggToSemiannual(userinfo_df, userinfo_df['source_parallel09'], userinfo_df['source_parallel0609'], userinfo_df['source_parallel0406'])
#userinfo_df['source_ignore'] = aggToSemiannual(userinfo_df, userinfo_df['source_ignore09'], userinfo_df['source_ignore0609'], userinfo_df['source_ignore0406'])
userinfo_df['status_ignore'] = aggToSemiannual(userinfo_df, userinfo_df['status_ignore09'], userinfo_df['status_ignore0609'], userinfo_df['status_ignore0406'])
userinfo_df['status_a'] = aggToSemiannual(userinfo_df, userinfo_df['status_a09'], userinfo_df['status_a0609'], userinfo_df['status_a0406'])
userinfo_df['status_b'] = aggToSemiannual(userinfo_df, userinfo_df['status_b09'], userinfo_df['status_b0609'], userinfo_df['status_b0406'])
userinfo_df['status_warn'] = aggToSemiannual(userinfo_df, userinfo_df['status_warn09'], userinfo_df['status_warn0609'], userinfo_df['status_warn0406'])
userinfo_df['status_ban'] = aggToSemiannual(userinfo_df, userinfo_df['status_ban09'], userinfo_df['status_ban0609'], userinfo_df['status_ban0406'])
userinfo_df['status_special_a'] = aggToSemiannual(userinfo_df, userinfo_df['status_special_a09'], userinfo_df['status_special_a0609'], userinfo_df['status_special_a0406'])
userinfo_df['status_child'] = aggToSemiannual(userinfo_df, userinfo_df['status_child09'], userinfo_df['status_child0609'], userinfo_df['status_child0406'])
userinfo_df['tap_porn'] = aggToSemiannual(userinfo_df, userinfo_df['tap_porn09'], userinfo_df['tap_porn0609'], userinfo_df['tap_porn0406'])
userinfo_df['tap_violencehorror'] = aggToSemiannual(userinfo_df, userinfo_df['tap_violencehorror09'], userinfo_df['tap_violencehorror0609'], userinfo_df['tap_violencehorror0406'])
userinfo_df['tap_alcoholgamble'] = aggToSemiannual(userinfo_df, userinfo_df['tap_alcoholgamble09'], userinfo_df['tap_alcoholgamble0609'], userinfo_df['tap_alcoholgamble0406'])
userinfo_df['tap_politicsreligion'] = aggToSemiannual(userinfo_df, userinfo_df['tap_politicsreligion09'], userinfo_df['tap_politicsreligion0609'], userinfo_df['tap_politicsreligion0406'])
userinfo_df['tap_meaningless'] = aggToSemiannual(userinfo_df, userinfo_df['tap_meaningless09'], userinfo_df['tap_meaningless0609'], userinfo_df['tap_meaningless0406'])
userinfo_df['tap_infringement'] = aggToSemiannual(userinfo_df, userinfo_df['tap_infringement09'], userinfo_df['tap_infringement0609'], userinfo_df['tap_infringement0406'])
userinfo_df['watch_punish_a'] = aggToSemiannual(userinfo_df, userinfo_df['watch_punish_a09'], userinfo_df['watch_punish_a0609'], userinfo_df['watch_punish_a0406'])
userinfo_df['watch_punish_b'] = aggToSemiannual(userinfo_df,userinfo_df['watch_punish_b09'], userinfo_df['watch_punish_b0609'], userinfo_df['watch_punish_b0406'])
userinfo_df['watch_punish_warn'] = aggToSemiannual(userinfo_df, userinfo_df['watch_punish_warn09'], userinfo_df['watch_punish_warn0609'], userinfo_df['watch_punish_warn0406'])
userinfo_df['watch_punish_special_a'] = aggToSemiannual(userinfo_df, userinfo_df['watch_punish_special_a09'], userinfo_df['watch_punish_special_a0609'], userinfo_df['watch_punish_special_a0406'])
userinfo_df['action_screen'] = aggToSemiannual(userinfo_df, userinfo_df['action_screen09'], userinfo_df['action_screen0609'], userinfo_df['action_screen0406'])
userinfo_df['rejection_cnt'] = aggToSemiannual(userinfo_df, userinfo_df['rejection_cnt09'], userinfo_df['rejection_cnt0609'], userinfo_df['rejection_cnt0406'])
userinfo_df['im_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['im_frequency09'], userinfo_df['im_frequency0609'], userinfo_df['im_frequency0406'])
userinfo_df['keyword_rejection_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['keyword_rejection_frequency09'], userinfo_df['keyword_rejection_frequency0609'], userinfo_df['keyword_rejection_frequency0406'])
userinfo_df['im_banned_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['im_banned_frequency09'], userinfo_df['im_banned_frequency0609'], userinfo_df['im_banned_frequency0406'])
userinfo_df['forum_rejection_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['forum_rejection_frequency09'], userinfo_df['forum_rejection_frequency0406'], userinfo_df['forum_rejection_frequency0609'])
userinfo_df['forum_post_cnt'] = aggToSemiannual(userinfo_df, userinfo_df['forum_post_cnt09'], userinfo_df['forum_post_cnt0609'], userinfo_df['forum_post_cnt0406'])
userinfo_df['forum_comment_cnt'] = aggToSemiannual(userinfo_df, userinfo_df['forum_comment_cnt09'], userinfo_df['forum_comment_cnt0609'], userinfo_df['forum_comment_cnt0406'])
userinfo_df['bar_banned_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['bar_banned_frequency09'], userinfo_df['bar_banned_frequency0609'], userinfo_df['bar_banned_frequency0406'])
userinfo_df['bar_deleted_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['bar_deleted_frequency09'], userinfo_df['bar_deleted_frequency0609'], userinfo_df['bar_deleted_frequency0406'])
userinfo_df['bar_ban_device_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['bar_ban_device_frequency09'], userinfo_df['bar_ban_device_frequency0609'], userinfo_df['bar_ban_device_frequency0406'])
userinfo_df['bar_ban_user_frequency'] = aggToSemiannual(userinfo_df, userinfo_df['bar_ban_user_frequency09'], userinfo_df['bar_ban_user_frequency0609'], userinfo_df['bar_ban_user_frequency0406'])


#%%
userinfo_df.info()
print(userinfo_df.head())

#%%
liveinfo_profile = liveinfo_df.profile_report(style={'full_width':True}, title = 'LiveInfo Profiling Report')
liveinfo_rejected_variables = liveinfo_profile.get_rejected_variables(threshold = 0.9)
liveinfo_profile.to_file(output_file = "liveinfo_output.html")

userinfo_profile = userinfo_df.profile_report(style={'full_width':True}, title = 'UserInfo Profiling Report')
userinfo_rejected_variables = userinfo_profile.get_rejected_variables(threshold = 0.9)
userinfo_profile.to_file(output_file = 'userinfo_output.html')

