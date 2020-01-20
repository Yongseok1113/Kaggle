#필요 모듈 호출
from pathlib import Path
import re
#from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, StratifiedKFold

import lightgbm as lgb
from sklearn.model_selection import train_test_split
 


#파일 경로지정
pd.set_option('display.max_colwidth',200)
pd.set_option('display.max_columns',None)

#파일 불러오기
def read_data():
    specs = pd.read_csv("specs.csv")
    train = pd.read_csv("train.csv", parse_dates=["timestamp"])
    labels = pd.read_csv("train_labels.csv")
    test = pd.read_csv('test.csv', parse_dates=["timestamp"])
    sample_submission = pd.read_csv('sample_submission.csv')
    return specs, train, labels, test

#train_labels 테이블에서 Assessment기준으로 world 세분화 (labels 자체는 assessment가 제출된 사람만 존재 => 4100)
def data_preprocess(train, test, labels):    
    t_f1 = labels.loc[:,['installation_id','title']]
    t_f2 = t_f1.title
    
    l1 = np.where(t_f2 == 'Mushroom Sorter (Assessment)', 'TREETOPCITY1',
                  np.where(t_f2 == 'Bird Measurer (Assessment)', 'TREETOPCITY2',
                           np.where(t_f2 == 'Cauldron Filler (Assessment)', 'MAGMAPEAK',
                                    np.where(t_f2 == 'Cart Balancer (Assessment)','CRYSTALCAVES1', 'CRYSTALCAVES2'))))
             
    pd.Series(l1).value_counts()
    t_f2.value_counts() #갯수확인 결과 제대로 분류됨
    t_f1['world2'] = l1 #installation_id, title, world
    
    #train 테이블에서 모든 title을  Assessment기준으로 world 세분화
    t1 = train.loc[:, ['installation_id','title','world']]
    t2 = t1.title
    
    l2 = np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Tree Top City - Level 1', 
                   t2=='Ordering Spheres'),  
                   t2=='All Star Sorting'),
                   t2=='Costume Box'),
                   t2=='Fireworks (Activity)'),  
                   t2=='12 Monkeys'),
                   t2=='Tree Top City - Level 2'), 
                   t2=='Flower Waterer (Activity)'),
                   t2=='Pirate\'s Tale'),
                   t2=='Mushroom Sorter (Assessment)'),
                 'TREETOPCITY1', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Air Show', 
                                                 t2=='Treasure Map'),
                                                 t2=='Tree Top City - Level 3'),
                                                 t2=='Crystals Rule'),
                                                 t2=='Rulers'),
                                                 t2=='Bug Measurer (Activity)'),
                                                 t2=='Bird Measurer (Assessment)'), 'TREETOPCITY2', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 =='Crystal Caves - Level 1',
                                                                t2=='Chow Time'), 
                                                                t2=='Balancing Act'),
                                                                t2=='Chicken Balancer (Activity)'),
                                                                t2=='Lifting Heavy Things'),
                                                                t2=='Crystal Caves - Level 2'),
                                                                t2=='Honey Cake'),
                                                                t2=='Happy Camel'),
                                                                t2=='Cart Balancer (Assessment)'), 'CRYSTALCAVES1', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Leaf Leader',
                                                                               t2=='Crystal Caves - Level 3'),
                                                                               t2=='Heavy, Heavier, Heaviest'),
                                                                               t2=='Pan Balance' ),
                                                                               t2=='Egg Dropper (Activity)'),
                                                                               t2=='Chest Sorter (Assessment)'), 'CRYSTALCAVES2', np.where(t2 == 'Welcome to Lost Lagoon!', np.nan, 'MAGMAPEAK' )))))
    
    train['world2'] = l2
    
    #train 테이블에서 사람별 Assessment를 수행한 행만 추출
    uk_name = t_f1[['installation_id','world2']].drop_duplicates()    
    
    feed_train = pd.merge(left=train, right=uk_name, on=['installation_id', 'world2'])


    #feed_train에서 평가를 제출한 것만 추출
    ass_table=feed_train.loc[feed_train['type']=='Assessment',:] 
    
    
    sub1 = ass_table.loc[(ass_table['title'] != "Bird Measurer (Assessment)") & (ass_table['event_code'] == 4100), :]
    sub2= ass_table.loc[(ass_table['title'] == "Bird Measurer (Assessment)") & (ass_table['event_code']==4110), :]
    
    s1= sub1[['game_session']].drop_duplicates()
    s2= sub2[['game_session']].drop_duplicates()
    
    submit = pd.concat([s1,s2], ignore_index=False)
    
    not_assessment= feed_train.loc[feed_train['type'] != 'Assessment', :]
    
    ass_submit = pd.merge(left= ass_table, right= submit, on='game_session', how='inner')
    
    real_train = pd.concat([not_assessment, ass_submit], ignore_index= True)
    real_train= real_train.sort_values(['installation_id','timestamp','event_count'])
    
    #test
    t1 = test.loc[:, ['installation_id','title','world']]
    t2 = t1.title
    
    l2 = np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Tree Top City - Level 1', 
                   t2=='Ordering Spheres'),  
                   t2=='All Star Sorting'),
                   t2=='Costume Box'),
                   t2=='Fireworks (Activity)'),  
                   t2=='12 Monkeys'),
                   t2=='Tree Top City - Level 2'), 
                   t2=='Flower Waterer (Activity)'),
                   t2=='Pirate\'s Tale'),
                   t2=='Mushroom Sorter (Assessment)'),
                 'TREETOPCITY1', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Air Show', 
                                                 t2=='Treasure Map'),
                                                 t2=='Tree Top City - Level 3'),
                                                 t2=='Crystals Rule'),
                                                 t2=='Rulers'),
                                                 t2=='Bug Measurer (Activity)'),
                                                 t2=='Bird Measurer (Assessment)'), 'TREETOPCITY2', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 =='Crystal Caves - Level 1',
                                                                t2=='Chow Time'), 
                                                                t2=='Balancing Act'),
                                                                t2=='Chicken Balancer (Activity)'),
                                                                t2=='Lifting Heavy Things'),
                                                                t2=='Crystal Caves - Level 2'),
                                                                t2=='Honey Cake'),
                                                                t2=='Happy Camel'),
                                                                t2=='Cart Balancer (Assessment)'), 'CRYSTALCAVES1', np.where(np.logical_or(np.logical_or(np.logical_or(np.logical_or(np.logical_or(t2 == 'Leaf Leader',
                                                                               t2=='Crystal Caves - Level 3'),
                                                                               t2=='Heavy, Heavier, Heaviest'),
                                                                               t2=='Pan Balance' ),
                                                                               t2=='Egg Dropper (Activity)'),
                                                                               t2=='Chest Sorter (Assessment)'), 'CRYSTALCAVES2', np.where(t2 == 'Welcome to Lost Lagoon!', np.nan, 'MAGMAPEAK' )))))
    
    test['world2'] = l2
         
    uk_id = test['installation_id'].drop_duplicates()
    
    df1 = pd.merge(left = test[test.type == 'Assessment'], right = uk_id, on=['installation_id'])
    df1 = df1.sort_values(by= ['installation_id', 'timestamp'])
    df2 = df1.installation_id.drop_duplicates(keep='last').index
    df3 = df1.loc[df2]
    
    real_test = pd.merge(left=test, right=df3.loc[:,['installation_id', 'world2']], on=['installation_id', 'world2'])

    return real_train, real_test

###############################################################################
#train 가공
#1.title count, mean, std
##count
def train_x1(real_train):
    event_enter= real_train.loc[real_train['event_code'] ==2000,:]
    
    game_table= event_enter.loc[event_enter['type'] != 'Assessment',['installation_id','timestamp','event_id','title','world2']]
    ass_table= event_enter.loc[event_enter['type']=='Assessment',['installation_id','game_session', 'timestamp','title','world2']]
    
    T_table = pd.merge(left =ass_table, right =game_table, on=['installation_id','world2'], how= 'left')
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y]
    
    t1 = T_table1.groupby(['installation_id','timestamp_x','game_session','title_x','world2','title_y']).event_id.count().reset_index()
    a1 = t1.columns.values
    a1[-1] = 'title_count'
    t1.columns = a1
    
    T_count = t1.pivot_table(values='title_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['title_y']).reset_index().fillna(0)
    train_x1 = T_count.drop(['installation_id','timestamp_x','title_x','world2'], axis=1)
    
    train_x1.columns.values[1:]= list(map(lambda x: x +'_cnt', list(train_x1.columns.values[1:])))
    
    ##mean
    T_count['hap'] = T_count['installation_id'] + '_' + T_count['title_x']
                
    table=pd.DataFrame()
    title_mean_table = pd.DataFrame()
    for i in T_count.groupby(T_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].mean()).T], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          title_mean_table = pd.concat([title_mean_table, df1], ignore_index=False)
    
    train_x1_1= title_mean_table.drop(['installation_id','timestamp_x','title_x','world2','hap'], axis=1)
    train_x1_1.columns.values[1:] = list(map(lambda x : x+'_mean', list(train_x1_1.columns.values[1: ])))
    
    ##std
    table=pd.DataFrame()
    title_std_table = pd.DataFrame()
    for i in T_count.groupby(T_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].std()).T.fillna(0)], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          title_std_table = pd.concat([title_std_table, df1], ignore_index=False)
    
    train_x1_2= title_std_table.drop(['installation_id','timestamp_x','title_x','world2','hap'], axis=1)
    train_x1_2.columns.values[1:] = list(map(lambda x : x+'_std', list(train_x1_2.columns.values[1: ])))
    return train_x1, train_x1_1, train_x1_2


#del T_count,T_table, T_table1, ass_table,event_enter, game_table, title_mean_table, title_std_table,t1


#2. event_code count,mean,std
def train_x2(real_train):
    game_table= real_train.loc[real_train['type'] != 'Assessment',['installation_id','timestamp','event_code','event_id','title','world2']]
    ass_table= real_train.loc[(real_train['type']=='Assessment') & (real_train['event_code']==2000),['installation_id','game_session', 'timestamp','title','world2']]
    
    T_table = pd.merge(left =ass_table, right = game_table, on = ['installation_id','world2'], how= 'left')
    
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y ]
    
    df_count = T_table1.groupby(['installation_id','game_session','timestamp_x','title_x','world2','event_code']).event_id.count().reset_index()
    a1 = df_count.columns.values
    a1[-1] = 'event_code_count'
    df_count.columns = a1
    
    EC_count = df_count.pivot_table(values='event_code_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['event_code']).reset_index().fillna(0)
    
    train_x2 = EC_count.drop(['installation_id','timestamp_x','title_x','world2'], axis=1)
    train_x2.columns.values[1:] = list(map(lambda x : str(int(x))+ '_cnt', list(train_x2.columns.values[1:])))
    
    ##mean
    EC_count['hap'] = EC_count['installation_id'] + '_' + EC_count['title_x']
                
    table=pd.DataFrame()
    code_mean_table = pd.DataFrame()
    for i in EC_count.groupby(EC_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].mean()).T], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          code_mean_table = pd.concat([code_mean_table, df1], ignore_index=False)
    
    train_x2_1= code_mean_table.drop(['installation_id','timestamp_x','title_x','world2','hap'], axis=1)
    train_x2_1.columns.values[1:] = list(map(lambda x : str(int(x))+ '_mean', list(train_x2_1.columns.values[1:])))
    
    ##std
    table=pd.DataFrame()
    code_std_table = pd.DataFrame()
    for i in EC_count.groupby(EC_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].std()).T.fillna(0)], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          code_std_table = pd.concat([code_std_table, df1], ignore_index=False)
    
    train_x2_2= code_std_table.drop(['installation_id','timestamp_x','title_x','world2','hap'], axis=1)
    train_x2_2.columns.values[1:] = list(map(lambda x : str(int(x))+ '_std', list(train_x2_2.columns.values[1:])))
    return train_x2, train_x2_1, train_x2_2


#del EC_count,T_table, T_table1, ass_table, game_table, code_mean_table, code_std_table, df_count


#3.title_event_code count
def train_x3(real_train):
    game_table= real_train.loc[real_train['type'] != 'Assessment',['installation_id','timestamp','event_code','event_id','title','world2']]
    ass_table= real_train.loc[(real_train['type']=='Assessment') & (real_train['event_code']==2000),['installation_id','game_session', 'timestamp','title','world2']]
    
    game_table['title_code'] = game_table['title'] +'_' + game_table['event_code'].astype('str')
    
    T_table = pd.merge(left =ass_table, right = game_table, on = ['installation_id','world2'], how= 'left')
    
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y ]
    
    df_count = T_table1.groupby(['installation_id','game_session','timestamp_x','title_x','world2','title_code']).event_id.count().reset_index()
    a1 = df_count.columns.values
    a1[-1] = 'title_code_count'
    df_count.columns = a1
    
    TEC_count = df_count.pivot_table(values='title_code_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['title_code']).reset_index().fillna(0)
    
    train_x3 = TEC_count.drop(['installation_id','timestamp_x','title_x','world2'], axis=1)
    train_x3.columns.values[1:] = list(map(lambda x : x+ '_cnt', list(train_x3.columns.values[1:])))
    return train_x3


#del ass_table, game_table, T_table, T_table1, TEC_count, df_count

#4. event_count cnt, mean, std
# 평가
def train_x4(real_train):
    assessment_table1 = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000), ['installation_id', 'game_session', 'timestamp', 'world2']]
    # 평가 외 컨텐츠
    act_table1 = real_train.loc[real_train.type != 'Assessment']
    act_table2 = act_table1.groupby('game_session').event_count.max().reset_index()
    # 소속별 평가이전 컨텐츠
    temp_table = pd.merge(left=act_table1, right = act_table2, on=['game_session', 'event_count'])
    temp_table2 = pd.merge(left=assessment_table1, right=temp_table, on=['installation_id', 'world2'], how='left')
    temp_table3 = temp_table2.loc[temp_table2.timestamp_x > temp_table2.timestamp_y]
    # max, mean, std
    eventcnt_max = temp_table3.groupby(['game_session_x', 'title'])[['event_count']].max().reset_index()
    eventcnt_max.title = eventcnt_max.title.map(lambda x : x + '_max')
    eventcnt_max = eventcnt_max.pivot_table(index='game_session_x', columns='title', values='event_count')
    
    eventcnt_mean = temp_table3.groupby(['game_session_x', 'title'])[['event_count']].mean().reset_index()
    eventcnt_mean.title = eventcnt_mean.title.map(lambda x : x + '_mean')
    eventcnt_mean = eventcnt_mean.pivot_table(index='game_session_x', columns='title', values='event_count')
    
    eventcnt_std = temp_table3.groupby(['game_session_x', 'title'])[['event_count']].std().reset_index().fillna(0)
    eventcnt_std.title = eventcnt_std.title.map(lambda x : x + '_std')
    eventcnt_std = eventcnt_std.pivot_table(index='game_session_x', columns='title', values='event_count')
    
    result_table = pd.concat([eventcnt_max, eventcnt_mean], axis=1)
    result_table = pd.concat([result_table, eventcnt_std], axis=1).reset_index().fillna(0)
    train_x4 = result_table.copy()
    a1 = train_x4.columns.values
    a1[0]= 'game_session'
    train_x4.columns =a1
    return train_x4

#5.accur 계산
def train_x5(real_train):
    train_assessment = real_train.loc[real_train.type == 'Assessment',:]
    
    # else(4100)
    train_a = train_assessment.loc[train_assessment.title != 'Bird Measurer (Assessment)',:]
    train_a_count = train_a.groupby(['installation_id','game_session','event_code'])['world2'].count().reset_index() 
    train_a_data = train_a.loc[train_a.event_code == 4100, ['installation_id','game_session','event_code','event_data']]
    
    
    pattern1 = re.compile('"correct":([a-z]+)',flags=re.IGNORECASE)
    train_a_data['t_f'] = train_a_data.event_data.str.findall(pattern1).str[0]
    train_a_data_true = train_a_data.loc[train_a_data.t_f == 'true',:]
    
    check1 = pd.merge(train_a_count, train_a_data_true, on=['installation_id','game_session','event_code'], how='left')
    
    
    # bird(4110)
    train_b = train_assessment.loc[train_assessment.title == 'Bird Measurer (Assessment)',:]
    train_b_count = train_b.groupby(['installation_id','game_session','event_code'])['world2'].count().reset_index()
    train_b_data = train_b.loc[train_b.event_code == 4110, ['installation_id','game_session','event_code','event_data']]
    
    pattern2 = re.compile('"correct":([a-z]+)',flags=re.IGNORECASE)
    train_b_data['t_f'] = train_b_data.event_data.str.findall(pattern2).str[0]
    train_b_data_true = train_b_data.loc[train_b_data.t_f == 'true',:]
    
    check2 = pd.merge(train_b_count, train_b_data_true, on=['installation_id','game_session','event_code'], how='left')
    
    
    # else
    #train_else_count = train_a_count.loc[train_a_count.event_code == 4100,:]
    train_else_count = check1.loc[check1.event_code == 4100,:]
    train_else = train_else_count.loc[:,['installation_id','game_session','event_code','world2','t_f']]
    train_else['correct'] = [1 if x == 'true' else 0 for x in train_else.t_f]
    train_else = train_else.loc[:,['installation_id','game_session','event_code','world2','correct']]
    
    # Bird Measurer (Assessment)
    train_bird_count = check2.loc[check2.event_code == 4110,:]
    train_bird = train_bird_count.loc[:,['installation_id','game_session','event_code','world2','t_f']]
    train_bird['correct'] = [1 if x == 'true' else 0 for x in train_bird.t_f]
    train_bird = train_bird.loc[:,['installation_id','game_session','event_code','world2','correct']]
    
    
    # combine
    train_acc = pd.concat([train_else,train_bird], ignore_index = True)
    train_acc.columns = ['installation_id', 'game_session', 'event_code', 'try', 'correct']
    train_acc = train_acc.loc[:,['installation_id','game_session','correct','try']]
    
    train_acc['accuracy'] = train_acc['correct']/train_acc['try']
    
    a_g = []
    for i in train_acc.accuracy:
        if i == 1 :
            a_g.append(3)
        elif i >= 0.5 :
            a_g.append(2)
        elif i > 0 :
            a_g.append(1)
        else :
            a_g.append(0)
    
    train_acc['accuracy_group'] = a_g
    
    train_x5 = train_acc.drop(['installation_id'],axis=1)
    return train_x5

#X6: 비디오 시청시간 점수화
def train_x6(real_train):
    feed_x6 = real_train.loc[:,['installation_id', 'game_session', 'timestamp', 'event_id', 'title', 'world2']]
    feed_x6 = feed_x6.sort_values(by=['installation_id', 'timestamp'])
    feed_x6 = feed_x6.set_index(['installation_id','timestamp']).reset_index() # set_index 종엽이 아이디어~ 
    #check = feed_x6.loc[(feed_x6.installation_id == '90db3a9f')]
    
    rule = {"Ordering Spheres":[1, 24, 61], "Costume Box":[1, 17, 61], "12 Monkeys":[1,14,109], "Pirate's Tale":[1,10,80], 
            "Treasure Map": [1,10, 156],    "Rulers":[1,25,126],       "Slop Problem":[1,7,60], "Balancing Act":[1,25,72], 
            "Lifting Heavy Things":[1,20,118], "Honey Cake":[1,70,142], "Heavy, Heavier, Heaviest":[1, 24, 61],
            "Welcome to Lost Lagoon!":[0, 0, 0], 
            "Tree Top City - Level 1":[0, 0, 0],
            "Tree Top City - Level 2":[0, 0, 0],
            "Tree Top City - Level 3":[0, 0, 0],
            "Magma Peak - Level 1":[0, 0, 0],
            "Magma Peak - Level 2":[0, 0, 0],
            "Crystal Caves - Level 1":[0, 0, 0],
            "Crystal Caves - Level 2":[0, 0, 0],
            "Crystal Caves - Level 3":[0, 0, 0],
                    }
    
    clip_list = ['Ordering Spheres', 'Costume Box', '12 Monkeys', "Pirate's Tale", 'Treasure Map', 'Rulers', 'Slop Problem', 'Balancing Act',
                 'Lifting Heavy Things', 'Honey Cake', 'Heavy, Heavier, Heaviest']
    
    # 클립 시간 측정 
    temp_duration = []
    temp_gamesession = []
    temp_installation_id = []
    #temp_installation_id_i = []
    temp_timestamp = []
    #temp_timestamp_i = []
    temp_world2 = []
    temp_title = []
    
    for i in range(0, len(feed_x6)):
        if(i == 0):
            print('시작')
        elif( feed_x6.event_id[i-1] == '27253bdc'):
            # i와 i-1이 서로다른 사람이 되는 경우  정렬을 했지만, i-1이 i보다 날짜가 커질 수 있음. 
            if( feed_x6.installation_id[i-1] == feed_x6.installation_id[i]):
                temp_duration.append((feed_x6.timestamp[i] - feed_x6.timestamp[i-1]).total_seconds())
            else:
                temp_duration.append(rule[feed_x6.title[i-1]][0])
            temp_world2.append(feed_x6.world2[i])    
            temp_gamesession.append(feed_x6.game_session[i])
            temp_installation_id.append(feed_x6.installation_id[i])
    #        temp_installation_id_i.append(feed_x6.installation_id[i-1])
            temp_timestamp.append(feed_x6.timestamp[i])
    #        temp_timestamp_i.append(feed_x6.timestamp[i-1])
            temp_title.append(feed_x6.title[i])
    
    df1 = pd.DataFrame()
    df1['installation_id'] = temp_installation_id
    df1['world2'] = temp_world2
    #df1['temp_installation_id_i'] = temp_installation_id_i
    df1['game_session'] = temp_gamesession
    df1['timestamp'] = temp_timestamp
    #df1['temp_timestamp_i'] = temp_timestamp_i
    df1['temp_duration'] = temp_duration
    df1['title'] = temp_title
    
    df2 = df1[df1.title.isin(clip_list)]
    df2.index = range(0, len(df2))
    
    
    # 점수 뽑기 
    score = []
    for i in range(0, len(df2)):
        # 바로나간 케이스
        if( df2.temp_duration[i] <= rule[df2.title[i]][0]):
            score.append(0)
        elif( df2.temp_duration[i] <= rule[df2.title[i]][1]):
            score.append(1)
        elif( df2.temp_duration[i] <= rule[df2.title[i]][2]):
            score.append(2)
        # 클립의 최대 사이즈보다 기간이 긴 경우는 중간에 앱을 끄거나 한 경우 인것 같으므로 안본것으로 간주 한다. 
        else:
            score.append(0)
    
    df2['score'] = score
    #df2.columns = ['installation_id', 'installation_id_before', 'game_session', 'temp_duration', 'timestamp', 'timestamp_before', 'title', 'score'] 
    
#    magma = real_train[(real_train.type == 'Clip') & (real_train.world2 == 'MAGMAPEAK')].title.unique()
#    treetop1 = real_train[(real_train.type == 'Clip') & (real_train.world2 == 'TREETOPCITY1')].title.unique()
#    treetop2 = real_train[(real_train.type == 'Clip') & (real_train.world2 == 'TREETOPCITY2')].title.unique()
#    crystal1 = real_train[(real_train.type == 'Clip') & (real_train.world2 == 'CRYSTALCAVES1')].title.unique()
#    crystal2 = real_train[(real_train.type == 'Clip') & (real_train.world2 == 'CRYSTALCAVES2')].title.unique()
    
    # 평가명을 key로 하고 평가에 소속된 클립들의 title이 value인 딕셔너리 
#    obj_dict = {'MAGMAPEAK': magma, 'TREETOPCITY1': treetop1, 'TREETOPCITY2': treetop2, 'CRYSTALCAVES1': crystal1, 'CRYSTALCAVES2': crystal2 }
    
    #평가별 진입 이벤트 id 리스트
#    start_assessment_id = ['3bfd1a65', 'f56e0afc', '7ad3efc6', '5b49460a', '90d848e0']
    
    df_Clip2 = df2.loc[:, ['installation_id', 'timestamp', 'title', 'score', 'world2']]
    df_Assessment2 = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000), ['installation_id','game_session', 'timestamp', 'world2']]
    df_Assessment2 = pd.merge(left=df_Assessment2, right=df_Clip2, on=['installation_id', 'world2'], how='left')
    
    df_Assessment2_3 = df_Assessment2.loc[df_Assessment2.timestamp_x > df_Assessment2.timestamp_y]
    df_Assessment2_3 = df_Assessment2_3.groupby(['game_session', 'title']).score.mean().reset_index()
    df_Assessment2_3 = df_Assessment2_3.pivot_table(index='game_session', columns='title', values='score').reset_index()
    
    df_Assessment2_1 = df_Assessment2.loc[df_Assessment2.timestamp_y.isna()]
    df_Assessment2_1 = pd.DataFrame(index = df_Assessment2_1.game_session, columns=df_Assessment2_3.columns[1:]).reset_index()
    
    train_x6 = pd.concat([df_Assessment2_1, df_Assessment2_3], ignore_index=True)
    train_x6 = train_x6.fillna(0)
    return train_x6

#X7 :Game별 game_session에서 최초 정클릭 까지의 count
def train_x7(real_train):
    dict_g = {'3bfd1a65' : ['1cc7cfca'],'f56e0afc' :[], '7ad3efc6': ['cfbd47c8','3d8c61b0'], '5b49460a' : ['2a444e03'], '90d848e0' :['792530f8']}
    dict_val = []
    for i in dict_g.values():
        for j in i:
            dict_val.append(j)
    
    train_x7_1 = pd.DataFrame()
    train_x7_1 =real_train.loc[(real_train['type'] == 'Assessment') & (real_train['event_code']==2000), ['game_session']].drop_duplicates()
    for i in range(0,len(dict_val)):
        G_table = real_train.loc[real_train.event_id ==dict_val[i], ['installation_id','timestamp','game_session','event_id','title','event_count','world2']]
        G_table1= G_table.groupby(['game_session','title','world2'])[['timestamp','event_count']].min().reset_index()
        G_table2 = G_table[['game_session','installation_id']].drop_duplicates()
        G_table3 = pd.merge(left = G_table1,right = G_table2, on= 'game_session', how= 'inner')
        T_g_table = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000)]
        T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp','world2']]
        T_g_table = pd.merge(left=T_g_table, right=G_table3, on=['installation_id','world2'], how='left')
        T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
        T_g_table_1 = T_g_table_1.loc[:, ['game_session_x']]
        T_g_table_1.columns = ['game_session']
        T_g_table_1['event_count'] = 0
        T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
        T_g_table_3 = T_g_table_3.groupby(['game_session_x'])[['event_count']].mean().reset_index()
        T_g_table_3.columns = ['game_session','event_count'] 
        train_x7 = pd.concat([T_g_table_1, T_g_table_3], ignore_index=True)
        train_x7_1 = pd.merge(left= train_x7_1, right= train_x7, on= 'game_session', how='left' )
    
    train_x7_1.columns.values[1:] = list(map(lambda x : 'x7_'+x , ['All Star Sorting', 'Chow Time','Happy Camel', 'Pan Balance', 'Dino Drink']))
    train_x7 = train_x7_1.copy()
    train_x7 = train_x7.fillna(train_x7.max())
    return train_x7

#X8 : Activity별 game_session에서 최초 정클릭까지의 count
def train_x8(real_train):
    real_train.loc[(real_train['type']=='Activity') & (real_train['event_code']==4030), ['title','event_id']].drop_duplicates()
    dict_a = {'3bfd1a65' : ['02a42007','5d042115'],'f56e0afc' :['e79f3763'], '7ad3efc6': ['56bcd38d'], '5b49460a' : [], '90d848e0' :['5e812b27','bb3e370b']}
    
    dict_val = []
    for i in dict_a.values():
        for j in i:
            dict_val.append(j)
    
    train_x8_1 = pd.DataFrame()
    train_x8_1 =real_train.loc[(real_train['type'] == 'Assessment') & (real_train['event_code']==2000), ['game_session']].drop_duplicates()
    for i in range(0,len(dict_val)):
        G_table = real_train.loc[real_train.event_id ==dict_val[i], ['installation_id','timestamp','game_session','event_id','title','event_count','world2']]
        G_table1= G_table.groupby(['game_session','title','world2'])[['timestamp','event_count']].min().reset_index()
        G_table2 = G_table[['game_session','installation_id']].drop_duplicates()
        G_table3 = pd.merge(left = G_table1,right = G_table2, on= 'game_session', how= 'inner')
        T_g_table = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000)]
        T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp','world2']]
        T_g_table = pd.merge(left=T_g_table, right=G_table3, on=['installation_id','world2'], how='left')
        T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
        T_g_table_1 = T_g_table_1.loc[:, ['game_session_x']]
        T_g_table_1.columns = ['game_session']
        T_g_table_1['event_count'] = 0
        T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
        T_g_table_3 = T_g_table_3.groupby(['game_session_x'])[['event_count']].mean().reset_index()
        T_g_table_3.columns = ['game_session','event_count'] 
        train_x8 = pd.concat([T_g_table_1, T_g_table_3], ignore_index=True)
        train_x8_1 = pd.merge(left= train_x8_1, right= train_x8, on= 'game_session', how='left' )
    
    
    train_x8_1.columns.values[1:] = list(map(lambda x : 'x8_'+x , ['Fireworks', 'Flower Waterer','Bug Measurer', 'Chicken Balancer', 'Sandcastle Builder','Bottle Filler']))
    train_x8 = train_x8_1.copy()
    train_x8 = train_x8.fillna(train_x8.max())
    return train_x8

#X9 : 사람별 평가별 시도 차수
def train_x9(real_train):
    dict_g = {'3bfd1a65' : ['1cc7cfca'],'f56e0afc' :[], '7ad3efc6': ['cfbd47c8','3d8c61b0'], '5b49460a' : ['2a444e03'], '90d848e0' :['792530f8']}
    a1 = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_id.isin(dict_g.keys()))]
    
    uk_inst = a1.installation_id.drop_duplicates()
    uk_title = a1.title.drop_duplicates()
    
    count = []
    for i in uk_inst :
        for j in uk_title :
            a2 = a1.loc[(a1.installation_id == i) & (a1.title == j),:]
            v1 = np.arange(0,a2.shape[0])
            for k in v1 :
                count.append(k)
    a1['count'] = count
    #a1.loc[:,['game_session','count']].to_csv(home/'x13.csv', index=False)
    train_x9 = a1.loc[:,['game_session','count']]
    return train_x9

#x10 : game help 클릭 수
def train_x10(real_train):
    dict_game_help = {'3bfd1a65' : ['6043a2b4'] ,   # mushroom sorter : treetop 1
                     'f56e0afc' : ['93edfe2e', '6f4bd64e'],               # bird measurer : treetop 2
                     '7ad3efc6' : ['19967db1', '05ad839b']  ,             # Cart balancer : crystal 1
                     '5b49460a' : ['e080a381', '67aa2ada'] ,              # chest sorter : crystal 2
                     '90d848e0' : ['6f8106d9', 'd3640339','6aeafed4', '92687c59']  # cauldron filler : Magmapeak
                     }
    
    
    dict_val = []
    for i in dict_game_help.values():
        for j in i:
            dict_val.append(j)
    
    
    G_table = real_train.loc[real_train.event_id.isin(dict_val), ['installation_id','timestamp','event_id', 'world2']]
    T_g_table = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000)]
    T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp', 'world2']]
    T_g_table = pd.merge(left=T_g_table, right=G_table, on=['installation_id', 'world2'], how='left')
    
    # game을 안한 경우
    T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
    T_g_table_1 = T_g_table_1.loc[:, ['game_session']]
    T_g_table_1['x10'] = 0
    
    
    T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
    T_g_table_3 = T_g_table_3.groupby(['game_session']).installation_id.count().reset_index()
    T_g_table_3.columns = ['game_session', 'x10']
    
    
    T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
    temp = T_g_table_3.groupby(['game_session']).installation_id.count().reset_index()
    temp.columns = ['game_session', 'count']
    T_g_table_3 = pd.merge(left=T_g_table_3, right=temp, on='game_session')
    T_g_table_3 = T_g_table_3.pivot_table(index='game_session', columns='event_id', values='count').reset_index()
    
    T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
    T_g_table_1 = pd.DataFrame(index = T_g_table_1.game_session, columns=T_g_table_3.columns[1:]).reset_index()
    
    train_x10 = pd.concat([T_g_table_3, T_g_table_1], ignore_index=True)
    train_x10.columns = ['game_session', 'Happy Camel', 'Chow Time', 'All Star Sorting', 'Leaf Leader', 'Bubble Bath', 'Air Show', 'Dino Drink', 'Scrub_A_Dub', 'Crystals Rule', 'Dino Dive', 'Pan Balancer']
    return train_x10

#x11 : Activity help 클릭 수
def train_x11(real_train):
    dict_act_help = {'3bfd1a65' : ['47f43a44','f54238ee'] ,   # mushroom sorter : treetop 1
                     'f56e0afc' : ['8d748b58'],               # bird measurer : treetop 2
                     '7ad3efc6' : ['85d1b0de']  ,             # Cart balancer : crystal 1
                     '5b49460a' : ['08ff79ad'] ,              # chest sorter : crystal 2
                     '90d848e0' : ['37937459', 'e7e44842','47efca07']  # cauldron filler : Magmapeak
                     }
    
    dict_val = []
    for i in dict_act_help.values():
        for j in i:
            dict_val.append(j)
    
    
    
    A_table = real_train.loc[real_train.event_id.isin(dict_val), ['installation_id','timestamp','event_id', 'world2']]
    T_a_table = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000)]
    T_a_table = T_a_table.loc[:, ['installation_id','game_session', 'timestamp', 'world2']]
    T_a_table = pd.merge(left=T_a_table, right=A_table, on=['installation_id', 'world2'], how='left')
    
    # game을 안한 경우
    T_a_table_1 = T_a_table.loc[T_a_table.timestamp_y.isna()]
    T_a_table_1 = T_a_table_1.loc[:, ['game_session']]
    T_a_table_1['x11'] = 0
    
    
    T_a_table_3 = T_a_table.loc[T_a_table.timestamp_x > T_a_table.timestamp_y]
    T_a_table_3 = T_a_table_3.groupby(['game_session']).installation_id.count().reset_index()
    T_a_table_3.columns = ['game_session', 'x11']
    
    
    T_a_table_3 = T_a_table.loc[T_a_table.timestamp_x > T_a_table.timestamp_y]
    temp = T_a_table_3.groupby(['game_session']).installation_id.count().reset_index()
    temp.columns = ['game_session', 'count']
    T_a_table_3 = pd.merge(left=T_a_table_3, right=temp, on='game_session')
    T_a_table_3 = T_a_table_3.pivot_table(index='game_session', columns='event_id', values='count').reset_index()
    
    T_a_table_1 = T_a_table.loc[T_a_table.timestamp_y.isna()]
    T_a_table_1 = pd.DataFrame(index = T_a_table_1.game_session, columns=T_a_table_3.columns[1:]).reset_index()
    
    train_x11 = pd.concat([T_a_table_3, T_a_table_1], ignore_index=True)
    train_x11.columns = ['game_session',
                         'Egg Dropper','Sandcastle Builder','Bottle Filler','Flower Waterer',
                         'Chicken Balancer','Bug Measurer','Watering Hole','Fireworks']
    return train_x11

#12. 요일, 시간
def train_x12(real_train):
    a_t = real_train.loc[(real_train.type=='Assessment') & (real_train.event_count == 1), :].loc[:,['game_session','timestamp']]
    a_t['hour']=[x.hour for x in a_t.timestamp]
    a_t['dayofweek']=[x.dayofweek for x in a_t.timestamp]
    a_t = a_t.drop(columns='timestamp')
    
    train_x12= a_t.copy()
    return train_x12

#13.
def train_x13(real_train):
    regex = re.compile('("x":[0-9]+).+("y":[0-9]+)')
    s1 = real_train.event_data.str.findall(regex).str[0]
    deleteindex = s1[s1.isna()].index
    click_train = real_train.drop(deleteindex)
    
    regex2 = re.compile('"x":([0-9]+)')
    s2 = click_train.event_data.str.findall(regex2).str[0]
    s2.name = 'point_x'
    
    regex3 = re.compile('"y":([0-9]+)')
    s3 = click_train.event_data.str.findall(regex3).str[0]  
    s3.name = 'point_y'
    
    click_train = pd.concat([click_train, s2], axis=1)
    click_train = pd.concat([click_train, s3], axis=1)
    click_train.point_x = click_train.point_x.astype(int)
    click_train.point_y = click_train.point_y.astype(int)
    
    assessment_table1 = real_train.loc[(real_train.type == 'Assessment') & (real_train.event_code == 2000), ['installation_id', 'game_session', 'timestamp', 'world2']]


    # 세션 내 클릭이 1개라 std하면 nan나오는 경우가 있음 
    click_train2 = click_train.loc[click_train.type != 'Assessment', ['game_session', 'point_x', 'point_y']]
    click_train2 = click_train2.groupby('game_session').std(ddof=0).reset_index()
    click_train2.columns = ['game_session', 'point_x_std', 'point_y_std']
    
    click_train3 = pd.merge(left=click_train2, right=click_train, on=['game_session']).drop_duplicates('game_session')
    
    click_train4 = pd.merge(left=assessment_table1, right=click_train3, on=['installation_id', 'world2'], how='left')
    click_train4 = click_train4.loc[click_train4.timestamp_x > click_train4.timestamp_y]
    
    click_train5 = click_train4.loc[:, ['game_session_x', 'title', 'point_x_std', 'point_y_std']].groupby(['game_session_x', 'title']).std(ddof=0).reset_index()
    click_train5['title_y'] = click_train5.title
    click_train5.title = click_train5.title.map(lambda x : x + '_x')
    click_train5.title_y = click_train5.title.map(lambda x : x + '_y')
    
    click_train5_x = click_train5.pivot_table(index='game_session_x', columns='title', values='point_x_std').fillna(0)
    click_train5_y = click_train5.pivot_table(index='game_session_x', columns='title_y', values='point_y_std').fillna(0)
    
    result = pd.concat([click_train5_x, click_train5_y], axis=1).reset_index()
    a1 = result.columns.values
    a1[0] = 'game_session'
    result.columns =a1
    
#    train_x13 = result.copy()
    return result

#14. 세션별 Ass - act(마지막활동) 시간
def train_x14(real_train):
    game_table= real_train.loc[real_train['type'] != 'Assessment',['installation_id','timestamp','event_count','title','world2']]
    ass_table= real_train.loc[(real_train['type']=='Assessment') & (real_train['event_code'] ==2000),['installation_id','game_session', 'timestamp','title','world2']]
    T_table = pd.merge(left =ass_table, right =game_table, on=['installation_id','world2'], how= 'left')
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y]
    
    # 세션별 Ass - act(clip,game,activity) 시간
    t1 = T_table1.set_index('installation_id')
    ass_time = t1.groupby(['installation_id','game_session']).timestamp_x.max().reset_index()
    last_act_time = t1.groupby(['installation_id','game_session']).timestamp_y.max().reset_index()
    ass_act_time = pd.merge(left=ass_time, right = last_act_time, on=['installation_id','game_session'])
    ass_act_time.columns = ['installation_id', 'game_session', 'Ass_time', 'Act_time']
    ass_act = ass_act_time['Ass_time'] - ass_act_time['Act_time']
    ass_act_time['Ass-Act_time'] = ass_act
    ass_act_time_final = ass_act_time.loc[:,['game_session','Ass-Act_time']]
    
    f1 = lambda x : x.total_seconds()
    seconds_time = pd.DataFrame(list(map(f1,ass_act_time_final['Ass-Act_time'])))
    ass_act_time_final = pd.concat([ass_act_time_final, seconds_time], axis=1)
    ass_act_time_final = ass_act_time_final.iloc[:,[0,2]]
    ass_act_time_final.columns = ['game_session', 'seconds']
    
    train_x14= ass_act_time_final.copy()

    return train_x14

#del df1, df2, df3, test, l2, uk_id,t1,t2
############################################################################
#test 가공
#1.title count, mean, std
def test_x1(real_test):
    ##count
    event_enter= real_test.loc[real_test['event_code'] ==2000,:]
    
    game_table= event_enter.loc[event_enter['type'] != 'Assessment',['installation_id','timestamp','event_id','title','world2']]
    ass_table= event_enter.loc[event_enter['type']=='Assessment',['installation_id','game_session', 'timestamp','title','world2']]
    
    T_table = pd.merge(left =ass_table, right =game_table, on=['installation_id','world2'], how= 'left')
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y]
    
    t1 = T_table1.groupby(['installation_id','timestamp_x','game_session','title_x','world2','title_y']).event_id.count().reset_index()
    a1 = t1.columns.values
    a1[-1] = 'title_count'
    t1.columns = a1
    
    T_count = t1.pivot_table(values='title_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['title_y']).reset_index().fillna(0)
    test_x1 = T_count.drop(['title_x','world2'], axis=1)
    
    test_x1.columns.values[3:]= list(map(lambda x: x +'_cnt', list(test_x1.columns.values[3:])))
    
    ##mean
    T_count['hap'] = T_count['installation_id'] + '_' + T_count['title_x']
                
    table=pd.DataFrame()
    title_mean_table = pd.DataFrame()
    for i in T_count.groupby(T_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].mean()).T], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          title_mean_table = pd.concat([title_mean_table, df1], ignore_index=False)
    
    test_x1_1= title_mean_table.drop(['title_x','world2','hap'], axis=1)
    test_x1_1.columns.values[3:] = list(map(lambda x : x+'_mean', list(test_x1_1.columns.values[3: ])))
    
    ##std
    table=pd.DataFrame()
    title_std_table = pd.DataFrame()
    for i in T_count.groupby(T_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].std()).T.fillna(0)], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          title_std_table = pd.concat([title_std_table, df1], ignore_index=False)
    
    test_x1_2= title_std_table.drop(['title_x','world2','hap'], axis=1)
    test_x1_2.columns.values[3:] = list(map(lambda x : x+'_std', list(test_x1_2.columns.values[3: ])))
    return test_x1, test_x1_1, test_x1_2

#2. event_code count,mean,std
def test_x2(real_test):
    game_table= real_test.loc[real_test['type'] != 'Assessment',['installation_id','timestamp','event_code','event_id','title','world2']]
    ass_table= real_test.loc[(real_test['type']=='Assessment') & (real_test['event_code']==2000),['installation_id','game_session', 'timestamp','title','world2']]
    
    T_table = pd.merge(left =ass_table, right = game_table, on = ['installation_id','world2'], how= 'left')
    
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y ]
    
    df_count = T_table1.groupby(['installation_id','game_session','timestamp_x','title_x','world2','event_code']).event_id.count().reset_index()
    a1 = df_count.columns.values
    a1[-1] = 'event_code_count'
    df_count.columns = a1
    
    EC_count = df_count.pivot_table(values='event_code_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['event_code']).reset_index().fillna(0)
    
    test_x2 = EC_count.drop(['title_x','world2'], axis=1)
    test_x2.columns.values[3:] = list(map(lambda x : str(int(x))+ '_cnt', list(test_x2.columns.values[3:])))
    
    ##mean
    EC_count['hap'] = EC_count['installation_id'] + '_' + EC_count['title_x']
                
    table=pd.DataFrame()
    code_mean_table = pd.DataFrame()
    for i in EC_count.groupby(EC_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].mean()).T], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          code_mean_table = pd.concat([code_mean_table, df1], ignore_index=False)
    
    test_x2_1= code_mean_table.drop(['title_x','world2','hap'], axis=1)
    test_x2_1.columns.values[3:] = list(map(lambda x : str(int(x))+ '_mean', list(test_x2_1.columns.values[3:])))
    
    ##std
    table=pd.DataFrame()
    code_std_table = pd.DataFrame()
    for i in EC_count.groupby(EC_count['hap']):
          df1=i[1]
          for i in range(0,len(df1)):
                table = pd.concat([table,pd.DataFrame(df1.iloc[0:(i+1),5:-1].std()).T.fillna(0)], ignore_index=False)
          df1.iloc[:,5:-1] = table.values
          table= pd.DataFrame()
          code_std_table = pd.concat([code_std_table, df1], ignore_index=False)
    
    test_x2_2= code_std_table.drop(['title_x','world2','hap'], axis=1)
    test_x2_2.columns.values[3:] = list(map(lambda x : str(int(x))+ '_std', list(test_x2_2.columns.values[3:])))
    return test_x2, test_x2_1, test_x2_2

#3.title_event_code count
def test_x3(real_test):
    game_table= real_test.loc[real_test['type'] != 'Assessment',['installation_id','timestamp','event_code','event_id','title','world2']]
    ass_table= real_test.loc[(real_test['type']=='Assessment') & (real_test['event_code']==2000),['installation_id','game_session', 'timestamp','title','world2']]
    
    game_table['title_code'] = game_table['title'] +'_' + game_table['event_code'].astype('str')
    
    T_table = pd.merge(left =ass_table, right = game_table, on = ['installation_id','world2'], how= 'left')
    
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y ]
    
    df_count = T_table1.groupby(['installation_id','game_session','timestamp_x','title_x','world2','title_code']).event_id.count().reset_index()
    a1 = df_count.columns.values
    a1[-1] = 'title_code_count'
    df_count.columns = a1
    
    TEC_count = df_count.pivot_table(values='title_code_count', index=['installation_id','timestamp_x','game_session','title_x','world2'], columns=['title_code']).reset_index().fillna(0)
    
    
    test_x3 = TEC_count.drop(['title_x','world2'], axis=1)
    test_x3.columns.values[3:] = list(map(lambda x : x+ '_cnt', list(test_x3.columns.values[3:])))
    return test_x3

#4. event_count cnt, mean, std
# 평가
def test_x4(real_test):
    assessment_table1 = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000), ['installation_id', 'game_session', 'timestamp', 'world2']]
    # 평가 외 컨텐츠
    act_table1 = real_test.loc[real_test.type != 'Assessment']
    act_table2 = act_table1.groupby('game_session').event_count.max().reset_index()
    # 소속별 평가이전 컨텐츠
    temp_table = pd.merge(left=act_table1, right = act_table2, on=['game_session', 'event_count'])
    temp_table2 = pd.merge(left=assessment_table1, right=temp_table, on=['installation_id', 'world2'], how='left')
    temp_table3 = temp_table2.loc[temp_table2.timestamp_x > temp_table2.timestamp_y]
    # max, mean, std
    eventcnt_max = temp_table3.groupby(['installation_id','timestamp_x','game_session_x', 'title'])[['event_count']].max().reset_index()
    eventcnt_max.title = eventcnt_max.title.map(lambda x : x + '_max')
    eventcnt_max = eventcnt_max.pivot_table(index=['installation_id','timestamp_x','game_session_x'], columns='title', values='event_count')
    
    eventcnt_mean = temp_table3.groupby(['installation_id','timestamp_x','game_session_x', 'title'])[['event_count']].mean().reset_index()
    eventcnt_mean.title = eventcnt_mean.title.map(lambda x : x + '_mean')
    eventcnt_mean = eventcnt_mean.pivot_table(index=['installation_id','timestamp_x','game_session_x'], columns='title', values='event_count')
    
    eventcnt_std = temp_table3.groupby(['installation_id','timestamp_x','game_session_x', 'title'])[['event_count']].std().reset_index().fillna(0)
    eventcnt_std.title = eventcnt_std.title.map(lambda x : x + '_std')
    eventcnt_std = eventcnt_std.pivot_table(index=['installation_id','timestamp_x','game_session_x'], columns='title', values='event_count')
    
    result_table = pd.concat([eventcnt_max, eventcnt_mean], axis=1)
    result_table = pd.concat([result_table, eventcnt_std], axis=1).reset_index().fillna(0)
    test_x4 = result_table.copy()
    
    a1 = test_x4.columns.values
    a1[2] = 'game_session'
    test_x4.columns =a1
    return test_x4

#X6: 비디오 시청시간 점수화
def test_x6(real_test):
    feed_x6 = real_test.loc[:,['installation_id', 'game_session', 'timestamp', 'event_id', 'title', 'world2']]
    feed_x6 = feed_x6.sort_values(by=['installation_id', 'timestamp'])
    feed_x6 = feed_x6.set_index(['installation_id','timestamp']).reset_index() # set_index 종엽이 아이디어~ 
    #check = feed_x6.loc[(feed_x6.installation_id == '90db3a9f')]
    
    rule = {"Ordering Spheres":[1, 24, 61], "Costume Box":[1, 17, 61], "12 Monkeys":[1,14,109], "Pirate's Tale":[1,10,80], 
            "Treasure Map": [1,10, 156],    "Rulers":[1,25,126],       "Slop Problem":[1,7,60], "Balancing Act":[1,25,72], 
            "Lifting Heavy Things":[1,20,118], "Honey Cake":[1,70,142], "Heavy, Heavier, Heaviest":[1, 24, 61],
            "Welcome to Lost Lagoon!":[0, 0, 0], 
            "Tree Top City - Level 1":[0, 0, 0],
            "Tree Top City - Level 2":[0, 0, 0],
            "Tree Top City - Level 3":[0, 0, 0],
            "Magma Peak - Level 1":[0, 0, 0],
            "Magma Peak - Level 2":[0, 0, 0],
            "Crystal Caves - Level 1":[0, 0, 0],
            "Crystal Caves - Level 2":[0, 0, 0],
            "Crystal Caves - Level 3":[0, 0, 0],
                    }
    
    clip_list = ['Ordering Spheres', 'Costume Box', '12 Monkeys', "Pirate's Tale", 'Treasure Map', 'Rulers', 'Slop Problem', 'Balancing Act',
                 'Lifting Heavy Things', 'Honey Cake', 'Heavy, Heavier, Heaviest']
    
    # 클립 시간 측정 
    temp_duration = []
    temp_gamesession = []
    temp_installation_id = []
    #temp_installation_id_i = []
    temp_timestamp = []
    #temp_timestamp_i = []
    temp_world2 = []
    temp_title = []
    
    for i in range(0, len(feed_x6)):
        if(i == 0):
            print('시작')
        elif( feed_x6.event_id[i-1] == '27253bdc'):
            # i와 i-1이 서로다른 사람이 되는 경우  정렬을 했지만, i-1이 i보다 날짜가 커질 수 있음. 
            if( feed_x6.installation_id[i-1] == feed_x6.installation_id[i]):
                temp_duration.append((feed_x6.timestamp[i] - feed_x6.timestamp[i-1]).total_seconds())
            else:
                temp_duration.append(rule[feed_x6.title[i-1]][0])
            temp_world2.append(feed_x6.world2[i])    
            temp_gamesession.append(feed_x6.game_session[i])
            temp_installation_id.append(feed_x6.installation_id[i])
    #        temp_installation_id_i.append(feed_x6.installation_id[i-1])
            temp_timestamp.append(feed_x6.timestamp[i])
    #        temp_timestamp_i.append(feed_x6.timestamp[i-1])
            temp_title.append(feed_x6.title[i])
    
    df1 = pd.DataFrame()
    df1['installation_id'] = temp_installation_id
    df1['world2'] = temp_world2
    #df1['temp_installation_id_i'] = temp_installation_id_i
    df1['game_session'] = temp_gamesession
    df1['timestamp'] = temp_timestamp
    #df1['temp_timestamp_i'] = temp_timestamp_i
    df1['temp_duration'] = temp_duration
    df1['title'] = temp_title
    
    df2 = df1[df1.title.isin(clip_list)]
    df2.index = range(0, len(df2))
    
    
    # 점수 뽑기 
    score = []
    for i in range(0, len(df2)):
        # 바로나간 케이스
        if( df2.temp_duration[i] <= rule[df2.title[i]][0]):
            score.append(0)
        elif( df2.temp_duration[i] <= rule[df2.title[i]][1]):
            score.append(1)
        elif( df2.temp_duration[i] <= rule[df2.title[i]][2]):
            score.append(2)
        # 클립의 최대 사이즈보다 기간이 긴 경우는 중간에 앱을 끄거나 한 경우 인것 같으므로 안본것으로 간주 한다. 
        else:
            score.append(0)
    
    df2['score'] = score
    #df2.columns = ['installation_id', 'installation_id_before', 'game_session', 'temp_duration', 'timestamp', 'timestamp_before', 'title', 'score'] 
    
#    magma = real_test[(real_test.type == 'Clip') & (real_test.world2 == 'MAGMAPEAK')].title.unique()
#    treetop1 = real_test[(real_test.type == 'Clip') & (real_test.world2 == 'TREETOPCITY1')].title.unique()
#    treetop2 = real_test[(real_test.type == 'Clip') & (real_test.world2 == 'TREETOPCITY2')].title.unique()
#    crystal1 = real_test[(real_test.type == 'Clip') & (real_test.world2 == 'CRYSTALCAVES1')].title.unique()
#    crystal2 = real_test[(real_test.type == 'Clip') & (real_test.world2 == 'CRYSTALCAVES2')].title.unique()
    
    # 평가명을 key로 하고 평가에 소속된 클립들의 title이 value인 딕셔너리 
#    obj_dict = {'MAGMAPEAK': magma, 'TREETOPCITY1': treetop1, 'TREETOPCITY2': treetop2, 'CRYSTALCAVES1': crystal1, 'CRYSTALCAVES2': crystal2 }
    
    #평가별 진입 이벤트 id 리스트
#    start_assessment_id = ['3bfd1a65', 'f56e0afc', '7ad3efc6', '5b49460a', '90d848e0']
    
    df_Clip2 = df2.loc[:, ['installation_id', 'timestamp', 'title', 'score', 'world2']]
    df_Assessment2 = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000), ['installation_id','game_session', 'timestamp', 'world2']]
    df_Assessment2 = pd.merge(left=df_Assessment2, right=df_Clip2, on=['installation_id', 'world2'], how='left')
    
    df_Assessment2_3 = df_Assessment2.loc[df_Assessment2.timestamp_x > df_Assessment2.timestamp_y]
    df_Assessment2_3 = df_Assessment2_3.groupby(['game_session', 'title']).score.mean().reset_index()
    df_Assessment2_3 = df_Assessment2_3.pivot_table(index='game_session', columns='title', values='score').reset_index()
    
    df_Assessment2_1 = df_Assessment2.loc[df_Assessment2.timestamp_y.isna()]
    df_Assessment2_1 = pd.DataFrame(index = df_Assessment2_1.game_session, columns=df_Assessment2_3.columns[1:]).reset_index()
    
    test_x6 = pd.concat([df_Assessment2_1, df_Assessment2_3], ignore_index=True)
    test_x6 = test_x6.fillna(0)
    return test_x6

#X7 :Game별 game_session에서 최초 정클릭 까지의 count
def test_x7(real_test):
    dict_g = {'3bfd1a65' : ['1cc7cfca'],'f56e0afc' :[], '7ad3efc6': ['cfbd47c8','3d8c61b0'], '5b49460a' : ['2a444e03'], '90d848e0' :['792530f8']}
    dict_val = []
    for i in dict_g.values():
        for j in i:
            dict_val.append(j)
    
    test_x7_1 = pd.DataFrame()
    test_x7_1 =real_test.loc[(real_test['type'] == 'Assessment') & (real_test['event_code']==2000), ['game_session']].drop_duplicates()
    for i in range(0,len(dict_val)):
        G_table = real_test.loc[real_test.event_id ==dict_val[i], ['installation_id','timestamp','game_session','event_id','title','event_count','world2']]
        G_table1= G_table.groupby(['game_session','title','world2'])[['timestamp','event_count']].min().reset_index()
        G_table2 = G_table[['game_session','installation_id']].drop_duplicates()
        G_table3 = pd.merge(left = G_table1,right = G_table2, on= 'game_session', how= 'inner')
        T_g_table = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000)]
        T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp','world2']]
        T_g_table = pd.merge(left=T_g_table, right=G_table3, on=['installation_id','world2'], how='left')
        T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
        T_g_table_1 = T_g_table_1.loc[:, ['game_session_x']]
        T_g_table_1.columns = ['game_session']
        T_g_table_1['event_count'] = 0
        T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
        T_g_table_3 = T_g_table_3.groupby(['game_session_x'])[['event_count']].mean().reset_index()
        T_g_table_3.columns = ['game_session','event_count'] 
        test_x7 = pd.concat([T_g_table_1, T_g_table_3], ignore_index=True)
        test_x7_1 = pd.merge(left= test_x7_1, right= test_x7, on= 'game_session', how='left' )
    
    test_x7_1.columns.values[1:] = list(map(lambda x : 'x7_'+x , ['All Star Sorting', 'Chow Time','Happy Camel', 'Pan Balance', 'Dino Drink']))
    test_x7 = test_x7_1.copy()
    test_x7 = test_x7.fillna(test_x7.max())
    return test_x7

#X8 : Activity별 game_session에서 최초 정클릭까지의 count
def test_x8(real_test):
    dict_a = {'3bfd1a65' : ['02a42007','5d042115'],'f56e0afc' :['e79f3763'], '7ad3efc6': ['56bcd38d'], '5b49460a' : [], '90d848e0' :['5e812b27','bb3e370b']}
    
    dict_val = []
    for i in dict_a.values():
        for j in i:
            dict_val.append(j)
    
    test_x8_1 = pd.DataFrame()
    test_x8_1 =real_test.loc[(real_test['type'] == 'Assessment') & (real_test['event_code']==2000), ['game_session']].drop_duplicates()
    for i in range(0,len(dict_val)):
        G_table = real_test.loc[real_test.event_id ==dict_val[i], ['installation_id','timestamp','game_session','event_id','title','event_count','world2']]
        G_table1= G_table.groupby(['game_session','title','world2'])[['timestamp','event_count']].min().reset_index()
        G_table2 = G_table[['game_session','installation_id']].drop_duplicates()
        G_table3 = pd.merge(left = G_table1,right = G_table2, on= 'game_session', how= 'inner')
        T_g_table = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000)]
        T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp','world2']]
        T_g_table = pd.merge(left=T_g_table, right=G_table3, on=['installation_id','world2'], how='left')
        T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
        T_g_table_1 = T_g_table_1.loc[:, ['game_session_x']]
        T_g_table_1.columns = ['game_session']
        T_g_table_1['event_count'] = 0
        T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
        T_g_table_3 = T_g_table_3.groupby(['game_session_x'])[['event_count']].mean().reset_index()
        T_g_table_3.columns = ['game_session','event_count'] 
        test_x8 = pd.concat([T_g_table_1, T_g_table_3], ignore_index=True)
        test_x8_1 = pd.merge(left= test_x8_1, right= test_x8, on= 'game_session', how='left' )
    
    
    test_x8_1.columns.values[1:] = list(map(lambda x : 'x8_'+x , ['Fireworks', 'Flower Waterer','Bug Measurer', 'Chicken Balancer', 'Sandcastle Builder','Bottle Filler']))
    test_x8 = test_x8_1.copy()
    test_x8 = test_x8.fillna(test_x8.max())
    return test_x8

#X9 : 사람별 평가별 시도 차수
def test_x9(real_test):
    dict_g = {'3bfd1a65' : ['1cc7cfca'],'f56e0afc' :[], '7ad3efc6': ['cfbd47c8','3d8c61b0'], '5b49460a' : ['2a444e03'], '90d848e0' :['792530f8']}
    a1 = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_id.isin(dict_g.keys()))]
    
    uk_inst = a1.installation_id.drop_duplicates()
    uk_title = a1.title.drop_duplicates()
    
    count = []
    for i in uk_inst :
        for j in uk_title :
            a2 = a1.loc[(a1.installation_id == i) & (a1.title == j),:]
            v1 = np.arange(0,a2.shape[0])
            for k in v1 :
                count.append(k)
    a1['count'] = count
    #a1.loc[:,['game_session','count']].to_csv(home/'x13.csv', index=False)
    test_x9 = a1.loc[:,['game_session','count']]
    return test_x9

#x10 : game help 클릭 수
def test_x10(real_test):
    dict_game_help = {'3bfd1a65' : ['6043a2b4'] ,   # mushroom sorter : treetop 1
                     'f56e0afc' : ['93edfe2e', '6f4bd64e'],               # bird measurer : treetop 2
                     '7ad3efc6' : ['19967db1', '05ad839b']  ,             # Cart balancer : crystal 1
                     '5b49460a' : ['e080a381', '67aa2ada'] ,              # chest sorter : crystal 2
                     '90d848e0' : ['6f8106d9', 'd3640339','6aeafed4', '92687c59']  # cauldron filler : Magmapeak
                     }
    
    
    dict_val = []
    for i in dict_game_help.values():
        for j in i:
            dict_val.append(j)
    
    
    G_table = real_test.loc[real_test.event_id.isin(dict_val), ['installation_id','timestamp','event_id', 'world2']]
    T_g_table = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000)]
    T_g_table = T_g_table.loc[:, ['installation_id','game_session', 'timestamp', 'world2']]
    T_g_table = pd.merge(left=T_g_table, right=G_table, on=['installation_id', 'world2'], how='left')
    
    # game을 안한 경우
    T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
    T_g_table_1 = T_g_table_1.loc[:, ['game_session']]
    T_g_table_1['x10'] = 0
    
    
    T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
    T_g_table_3 = T_g_table_3.groupby(['game_session']).installation_id.count().reset_index()
    T_g_table_3.columns = ['game_session', 'x10']
    
    
    T_g_table_3 = T_g_table.loc[T_g_table.timestamp_x > T_g_table.timestamp_y]
    temp = T_g_table_3.groupby(['game_session']).installation_id.count().reset_index()
    temp.columns = ['game_session', 'count']
    T_g_table_3 = pd.merge(left=T_g_table_3, right=temp, on='game_session')
    T_g_table_3 = T_g_table_3.pivot_table(index='game_session', columns='event_id', values='count').reset_index()
    
    T_g_table_1 = T_g_table.loc[T_g_table.timestamp_y.isna()]
    T_g_table_1 = pd.DataFrame(index = T_g_table_1.game_session, columns=T_g_table_3.columns[1:]).reset_index()
    
    test_x10 = pd.concat([T_g_table_3, T_g_table_1], ignore_index=True)
    test_x10.columns = ['game_session', 'Happy Camel', 'Chow Time', 'All Star Sorting', 'Leaf Leader', 'Bubble Bath', 'Air Show', 'Dino Drink', 'Scrub_A_Dub', 'Crystals Rule', 'Dino Dive', 'Pan Balancer']
    return test_x10

#del G_table, T_g_table,T_g_table_1,T_g_table_3, dict_val

#x11 : Activity help 클릭 수
def test_x11(real_test):
    dict_act_help = {'3bfd1a65' : ['47f43a44','f54238ee'] ,   # mushroom sorter : treetop 1
                     'f56e0afc' : ['8d748b58'],               # bird measurer : treetop 2
                     '7ad3efc6' : ['85d1b0de']  ,             # Cart balancer : crystal 1
                     '5b49460a' : ['08ff79ad'] ,              # chest sorter : crystal 2
                     '90d848e0' : ['37937459', 'e7e44842','47efca07']  # cauldron filler : Magmapeak
                     }
    
    dict_val = []
    for i in dict_act_help.values():
        for j in i:
            dict_val.append(j)
    
    
    
    A_table = real_test.loc[real_test.event_id.isin(dict_val), ['installation_id','timestamp','event_id', 'world2']]
    T_a_table = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000)]
    T_a_table = T_a_table.loc[:, ['installation_id','game_session', 'timestamp', 'world2']]
    T_a_table = pd.merge(left=T_a_table, right=A_table, on=['installation_id', 'world2'], how='left')
    
    # game을 안한 경우
    T_a_table_1 = T_a_table.loc[T_a_table.timestamp_y.isna()]
    T_a_table_1 = T_a_table_1.loc[:, ['game_session']]
    T_a_table_1['x11'] = 0
    
    
    T_a_table_3 = T_a_table.loc[T_a_table.timestamp_x > T_a_table.timestamp_y]
    T_a_table_3 = T_a_table_3.groupby(['game_session']).installation_id.count().reset_index()
    T_a_table_3.columns = ['game_session', 'x11']
    
    
    T_a_table_3 = T_a_table.loc[T_a_table.timestamp_x > T_a_table.timestamp_y]
    temp = T_a_table_3.groupby(['game_session']).installation_id.count().reset_index()
    temp.columns = ['game_session', 'count']
    T_a_table_3 = pd.merge(left=T_a_table_3, right=temp, on='game_session')
    T_a_table_3 = T_a_table_3.pivot_table(index='game_session', columns='event_id', values='count').reset_index()
    
    T_a_table_1 = T_a_table.loc[T_a_table.timestamp_y.isna()]
    T_a_table_1 = pd.DataFrame(index = T_a_table_1.game_session, columns=T_a_table_3.columns[1:]).reset_index()
    
    test_x11 = pd.concat([T_a_table_3, T_a_table_1], ignore_index=True)
    test_x11.columns = ['game_session',
                         'Egg Dropper','Sandcastle Builder','Bottle Filler','Flower Waterer',
                         'Chicken Balancer','Bug Measurer','Watering Hole','Fireworks']
    return test_x11

#12. 요일, 시간
def test_x12(real_test):
    a_t = real_test.loc[(real_test.type=='Assessment') & (real_test.event_count == 1), :].loc[:,['game_session','timestamp']]
    a_t['hour']=[x.hour for x in a_t.timestamp]
    a_t['dayofweek']=[x.dayofweek for x in a_t.timestamp]
    a_t = a_t.drop(columns='timestamp')
    
    test_x12= a_t.copy()
    return test_x12

#13.
def test_x13(real_test):    
    regex = re.compile('("x":[0-9]+).+("y":[0-9]+)')
    s1 = real_test.event_data.str.findall(regex).str[0]
    deleteindex = s1[s1.isna()].index
    click_test = real_test.drop(deleteindex)
    
    regex2 = re.compile('"x":([0-9]+)')
    s2 = click_test.event_data.str.findall(regex2).str[0]
    s2.name = 'point_x'
    
    regex3 = re.compile('"y":([0-9]+)')
    s3 = click_test.event_data.str.findall(regex3).str[0]  
    s3.name = 'point_y'
    
    click_test = pd.concat([click_test, s2], axis=1)
    click_test = pd.concat([click_test, s3], axis=1)
    click_test.point_x = click_test.point_x.astype(int)
    click_test.point_y = click_test.point_y.astype(int)
    
    assessment_table1 = real_test.loc[(real_test.type == 'Assessment') & (real_test.event_code == 2000), ['installation_id', 'game_session', 'timestamp', 'world2']]
    
    # 세션 내 클릭이 1개라 std하면 nan나오는 경우가 있음 
    click_test2 = click_test.loc[click_test.type != 'Assessment', ['game_session', 'point_x', 'point_y']]
    click_test2 = click_test2.groupby('game_session').std(ddof=0).reset_index()
    click_test2.columns = ['game_session', 'point_x_std', 'point_y_std']
    
    click_test3 = pd.merge(left=click_test2, right=click_test, on=['game_session']).drop_duplicates('game_session')
    
    click_test4 = pd.merge(left=assessment_table1, right=click_test3, on=['installation_id', 'world2'], how='left')
    click_test4 = click_test4.loc[click_test4.timestamp_x > click_test4.timestamp_y]
    
    click_test5 = click_test4.loc[:, ['game_session_x', 'title', 'point_x_std', 'point_y_std']].groupby(['game_session_x', 'title']).std(ddof=0).reset_index()
    click_test5['title_y'] = click_test5.title
    click_test5.title = click_test5.title.map(lambda x : x + '_x')
    click_test5.title_y = click_test5.title.map(lambda x : x + '_y')
    
    click_test5_x = click_test5.pivot_table(index='game_session_x', columns='title', values='point_x_std').fillna(0)
    click_test5_y = click_test5.pivot_table(index='game_session_x', columns='title_y', values='point_y_std').fillna(0)
    
    result = pd.concat([click_test5_x, click_test5_y], axis=1).reset_index()
    a1 = result.columns.values
    a1[0] = 'game_session'
    result.columns =a1
#    test_x13 = result.copy()
    return result

#14. 세션별 Ass - act(마지막활동) 시간
def test_x14(real_test):
    game_table= real_test.loc[real_test['type'] != 'Assessment',['installation_id','timestamp','event_count','title','world2']]
    ass_table= real_test.loc[(real_test['type']=='Assessment') & (real_test['event_code'] ==2000),['installation_id','game_session', 'timestamp','title','world2']]
    T_table = pd.merge(left =ass_table, right =game_table, on=['installation_id','world2'], how= 'left')
    T_table1 = T_table.loc[T_table.timestamp_x > T_table.timestamp_y]
    
    # 세션별 Ass - act(clip,game,activity) 시간
    t1 = T_table1.set_index('installation_id')
    ass_time = t1.groupby(['installation_id','game_session']).timestamp_x.max().reset_index()
    last_act_time = t1.groupby(['installation_id','game_session']).timestamp_y.max().reset_index()
    ass_act_time = pd.merge(left=ass_time, right = last_act_time, on=['installation_id','game_session'])
    ass_act_time.columns = ['installation_id', 'game_session', 'Ass_time', 'Act_time']
    ass_act = ass_act_time['Ass_time'] - ass_act_time['Act_time']
    ass_act_time['Ass-Act_time'] = ass_act
    ass_act_time_final = ass_act_time.loc[:,['game_session','Ass-Act_time']]
    
    f1 = lambda x : x.total_seconds()
    seconds_time = pd.DataFrame(list(map(f1,ass_act_time_final['Ass-Act_time'])))
    ass_act_time_final = pd.concat([ass_act_time_final, seconds_time], axis=1)
    ass_act_time_final = ass_act_time_final.iloc[:,[0,2]]
    ass_act_time_final.columns = ['game_session', 'seconds']
    
    test_x14= ass_act_time_final.copy()
    return test_x14

# =============================================================================
# main
# =============================================================================
specs, train, labels, test = read_data()
print('Data load complete. time: {}'.format(datetime.now()))
real_train, real_test = data_preprocess(train, test, labels)

train_x1, train_x1_1, train_x1_2 = train_x1(real_train)
train_x2, train_x2_1, train_x2_2 = train_x2(real_train)
train_x3 = train_x3(real_train)
train_x4 = train_x4(real_train)
train_x5 = train_x5(real_train)
train_x6 = train_x6(real_train)
train_x7 = train_x7(real_train)
train_x8 = train_x8(real_train)
train_x9 = train_x9(real_train)
train_x10 = train_x10(real_train)
train_x11 = train_x11(real_train)
train_x12 = train_x12(real_train)
train_x13 = train_x13(real_train)
train_x14 = train_x14(real_train)

#train 합치기
feed_train = real_train.loc[(real_train['type']=='Assessment') & (real_train['event_code']==2000), ['installation_id', 'game_session']]

feed_train= pd.merge(left =feed_train, right = train_x1, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x1_1, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x1_2, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x2, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x2_1, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x2_2, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x3, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x4, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x5, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x6, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x7, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x8, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x9, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x10, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x11, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x12, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x13, on='game_session', how='left')
feed_train= pd.merge(left =feed_train, right = train_x14, on='game_session', how='left')

feed_train = feed_train.fillna(0)
feed_train = feed_train.drop(['correct','try','accuracy'],axis=1)
feed_train = feed_train.sort_values('installation_id')

#test 
test_x1, test_x1_1, test_x1_2 = test_x1(real_test)
test_x2, test_x2_1, test_x2_2 = test_x2(real_test)
test_x3 = test_x3(real_test)
test_x4 = test_x4(real_test)
#train_x5 = train_x5(real_train)
test_x6 = test_x6(real_test)
test_x7 = test_x7(real_test)
test_x8 = test_x8(real_test)
test_x9 = test_x9(real_test)
test_x10 = test_x10(real_test)
test_x11 = test_x11(real_test)
test_x12 = test_x12(real_test)
test_x13 = test_x13(real_test)
test_x14 = test_x14(real_test)

#test 합치기
feed_test = real_test.loc[(real_test['type']=='Assessment') & (real_test['event_code']==2000), ['installation_id','timestamp','game_session']]
a1 = feed_test.columns.values
a1[1] = 'timestamp_x'
feed_test.columns = a1

feed_test= pd.merge(left =feed_test, right = test_x1, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x1_1, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x1_2, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x2, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x2_1, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x2_2, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x3, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x4, on=['installation_id','timestamp_x','game_session'], how='left')
feed_test= pd.merge(left =feed_test, right = test_x6, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x7, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x8, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x9, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x10, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x11, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x12, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x13, on='game_session', how='left')
feed_test= pd.merge(left =feed_test, right = test_x14, on='game_session', how='left')


feed_test = feed_test.fillna(0)

last_index= feed_test.installation_id.drop_duplicates(keep='last').index
feed_test  = feed_test.loc[last_index,:]
feed_test = feed_test.drop(['timestamp_x'], axis=1)

train_col = feed_train.columns.values
test_col= feed_test.columns.values

l1 = []
for i in list(train_col) :
      if i not in list(test_col):
            l1.append(i)


for i in l1:
      feed_test[str(i)]=0

feed_test = feed_test.drop(['accuracy_group'],axis=1)
feed_test = feed_test.sort_values('installation_id')

######################################################################
#model
#from sklearn.model_selection import train_test_split

# train accuracy_group 제외

from sklearn.preprocessing import RobustScaler
robustScaler = RobustScaler()
robustScaler.fit(feed_train.drop(['installation_id', 'game_session', 'accuracy_group'], axis=1))
train_data_robustScaled = robustScaler.transform(feed_train.drop(['installation_id', 'game_session', 'accuracy_group'], axis=1))
test_data_robustScaled = robustScaler.transform(feed_test.drop(['installation_id', 'game_session'], axis=1))

x_train = train_data_robustScaled
x_test = test_data_robustScaled
y_train = pd.DataFrame(feed_train.accuracy_group)

train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

train_ds = lgb.Dataset(train_x, label = train_y)
test_ds = lgb.Dataset(test_x, label = test_y)

params = {'learning_rate': 0.01, 
          'n_estimators':5000,
          'max_depth': -1, 
          'boosting_type': 'gbdt', 
          'objective': 'multiclass', 
          'num_class':4,
          'metric': 'multi_logloss', 
#          'num_leaves': 144, 
          'feature_fraction': 0.9, 
          'early_stopping_rounds': 100}

model = lgb.train(params, train_ds, 10000, test_ds, verbose_eval=100, early_stopping_rounds=100)

predict_test = model.predict(x_test, num_iteration=model.best_iteration)
loss_score = cohen_kappa_score(train_y, predict_train.argmax(axis=1), weights = 'quadratic')

def predict(sample_sumission, y_pred):
    sample_submission['accuracy_group'] = y_pred.argmax(axis=1)
    sample_submission.to_csv('submission.csv', index=False)
    print(sample_submission['accuracy_group'].value_counts(normalize=True))
    
predict(sample_submission, predict_test)

# 다른방식
feed_train_1 = feed_train.drop('game_session', axis=1)
feed_test_1 = feed_test.drop(['game_session', 'installation_id'], axis=1)
kf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 42)
target = 'accuracy_group'
oof_pred = np.zeros((len(feed_train_1), 4))
y_pred = np.zeros((len(feed_test_1), 4))
for fold, (tr_ind, val_ind) in enumerate(kf.split(feed_train_1, feed_train_1[target])):
    print('Fold {}'.format(fold + 1))
    x_train, x_val = feed_train_1.iloc[tr_ind], feed_train_1.iloc[val_ind]
    y_train, y_val = feed_train_1[target][tr_ind], feed_train_1[target][val_ind]
    
    robustScaler = RobustScaler()
    robustScaler.fit(x_train)
    train_data_robustScaled = robustScaler.transform(x_train)
    test_data_robustScaled = robustScaler.transform(x_val)

    
    train_set = lgb.Dataset(train_data_robustScaled, y_train)
    val_set = lgb.Dataset(test_data_robustScaled, y_val)

    params = {
        'learning_rate': 0.07,
        'metric': 'multi_logloss',
        'objective': 'multiclass',
        'num_classes': 4,
        'feature_fraction': 0.7,
        'subsample': 0.7,
        'n_jobs': -1,
        'seed': 50,
        'max_depth': 10
    }

    model = lgb.train(params, train_set, num_boost_round = 1000000, early_stopping_rounds = 100, 
                      valid_sets=[train_set, val_set], verbose_eval = 100)
    oof_pred[val_ind] = model.predict(x_val)
    y_pred += model.predict(test_data_robustScaled) / 10
loss_score = cohen_kappa_score(feed_train_1[target], np.argmax(oof_pred, axis = 1), weights = 'quadratic')
result = pd.Series(np.argmax(oof_pred, axis = 1))
print('Our oof cohen kappa score is: ', loss_score)
print(result.value_counts(normalize = True))

predict(sample_submission, predict_test)