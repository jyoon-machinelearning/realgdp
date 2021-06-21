# This python file is made to generate the result of the following paper:
# Yoon, J. Forecasting of Real GDP Growth Using Machine Learning Models: Gradient Boosting and Random Forest Approach. Comput Econ 57, 247–265 (2021). https://doi.org/10.1007/s10614-020-10054-w

# If you want to cite this work, please cite as follows:
# Yoon, J. Forecasting of Real GDP Growth Using Machine Learning Models: Gradient Boosting and Random Forest Approach. Comput Econ 57, 247–265 (2021). https://doi.org/10.1007/s10614-020-10054-w

# If you have any question, please contact via email at j.yoon@aoni.waseda.jp

import numpy as np
import pandas as pd
import openpyxl 
import time
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor



path=r'rawdata.xlsx' #path of the rawdata file ex) D:\folder\rawdata.xlsx
path2=r'benchmark.xlsx' #path of the benchmark file ex) D:\folder\benchmark.xlsx
save_path = r''  #path of the final result file to be stored ex) D:\folder\

gdpg=pd.read_excel(path, sheet_name='quarter')

startyear=2001
testyear=2001
lastyear=2018
qahead=2
rangelast=int(gdpg['year'][len(gdpg)-1])-testyear+1

b=[]
for h in range(0,lastyear-testyear+1):
    b.append(h)
print(b)

writer=pd.DataFrame(index=b)
writer['year']=0
writer['actual rgdpg']=0
writer['KNN']=0
writer['RF']=0
writer['boosting']=0
writer['gradboosting']=0
writer['bagging']=0
writer['elasticnet']=0
writer['rnn']=0
writer['imf']=0
writer['boj']=0
writer['rmse_knn']=0
writer['rmse_RF']=0
writer['rmse_boosting']=0
writer['rmse_bagging']=0
writer['rmse_elastic']=0
writer['rmse_imf_f']=0
writer['rmse_boj_f']=0
writer['rmse_imf_s']=0
writer['rmse_boj_s']=0
writer['rmse_gradboosting']=0
writer['rf.best_params']=0
writer['gb.best_params']=0
writer['en.best_params']=0
writer['rf.best_params_internal']=0
writer['gb.best_params_internal']=0
writer['RF_internal_1q_2q_yoy']=0
writer['GB_internal_1q_2q_yoy']=0
writer['tscv2']=0

for i in range(1, rangelast):
    gdpg=pd.read_excel(path, sheet_name='quarter')
    benchmark=pd.read_excel(path2, sheet_name='benchmark')
    
    gdpg['quarter']=gdpg['quarter'].replace('Qtr ', '',regex=True)
    gdpg['gdpg_a_e_1q']=gdpg['gdpg_a'].shift(-1)
    
    
    if qahead==1:
        gdpg=gdpg.drop(['gdpg_a_e_3q','gdpg_a_e_2q'], axis=1)
        target='gdpg_a_e_1q'   
    if qahead==2:
        gdpg=gdpg.drop(['gdpg_a_e_3q','gdpg_a_e_1q'], axis=1)
        target='gdpg_a_e_2q'
    if qahead==3:
        gdpg=gdpg.drop(['gdpg_a_e_1q','gdpg_a_e_2q'], axis=1)
        target='gdpg_a_e_3q'       
    
    book = openpyxl.load_workbook(path)
    sheet = book.active
 

    testyear2=testyear-1+i
    row_count = len(gdpg)-((len(gdpg)/4)-(testyear2-gdpg['year'][0]))*4-2
    print(row_count)
    
    row_count=int(row_count)-(qahead-2)-(qahead-1)
    row_count2a=row_count+(qahead-1)
    row_count_test_a=row_count
    row_count2_a=row_count_test_a+1
    
    row_count_test=row_count+(qahead-1)
    row_count2=row_count_test+1


    X=gdpg.drop([target,'year','quarter'], axis=1)
    
    feature_names=X.columns
       
    y=gdpg[target]
    print(gdpg[0:row_count])
    print(gdpg[row_count_test_a:row_count2_a])
    
    print(gdpg[0:row_count2a])
    print(gdpg[row_count_test:row_count2])
    
   
    X_train = X[0:row_count]
    X_test = X[row_count_test_a:row_count2_a]
    y_train=gdpg[target][0:row_count]
    y_test = gdpg[target][row_count_test_a:row_count2_a]  

     
# =============================================================================
  
    k=i-1
    rs=0
   
    n_samples = row_count+1
    n_splits = 10
    n_folds = n_splits + 1
    indices = np.arange(n_samples)
    test_size = (n_samples // n_folds)
    test_starts = range(test_size + n_samples % n_folds, n_samples, test_size)

    tscv = [(np.arange(0,j+test_size-2,1),np.arange(j+test_size-2,j+test_size-1,1)) for j in range(test_size + n_samples % n_folds, n_samples, test_size)]
    
    print(tscv)
    parameters_rf = {'n_estimators': (100,500,1000), 'max_depth':(1,3,5,7,9,11,13,15,17,19)}
    parameters_gb = {'n_estimators': (100,500,1000), 'learning_rate':(0.3,0.1,0.01,0.001,0.0001),'max_depth':(1,3,5,7,9,11,13,15,17,19)}

    grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=rs), param_grid=parameters_rf,cv=tscv, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X_train, y_train)
    print (grid_search_rf.best_params_)
    rf_para=str(grid_search_rf.best_params_)
    max_depth_rf=float(rf_para.split("{'max_depth': ", 1)[1].split(",", 1)[0])

    n_estimators_rf=int(rf_para.split("n_estimators': ", 1)[1].split("}", 1)[0])
    RF=RandomForestRegressor(n_estimators=n_estimators_rf,max_depth=max_depth_rf, random_state=rs)
    RF.fit(X_train, y_train)
    
    
    grid_search_gb = GridSearchCV(estimator=GradientBoostingRegressor(random_state=rs), param_grid=parameters_gb, cv=tscv, scoring='neg_mean_squared_error')
    grid_search_gb.fit(X_train, y_train)
    print (grid_search_gb.best_params_)
    gb_para=str(grid_search_gb.best_params_)
    learning_rate_gb=float(gb_para.split("{'learning_rate': ", 1)[1].split(",", 1)[0])
    n_estimators_gb=int(gb_para.split("n_estimators': ", 1)[1].split("}", 1)[0])
    max_depth_gb=float(gb_para.split("max_depth': ", 1)[1].split(",", 1)[0])

    gradboosting=GradientBoostingRegressor(n_estimators=n_estimators_gb, learning_rate=learning_rate_gb, random_state=rs, max_depth=max_depth_gb)
    gradboosting.fit(X_train, y_train) 


    writer['rf.best_params_internal'].iloc[k]=str(grid_search_rf.best_params_)
    writer['gb.best_params_internal'].iloc[k]=str(grid_search_gb.best_params_)

    writer['RF_internal_1q_2q_yoy'].iloc[k]=RF.predict(X_test)
    writer['GB_internal_1q_2q_yoy'].iloc[k]=gradboosting.predict(X_test)


# =============================================================================

    target_gb=target+'_gb'
    target_rf=target+'_rf'

    gdpg[target_gb]=gdpg[target]
    gdpg[target_rf]=gdpg[target]   
    gdpg[target_gb][row_count_test_a]=gradboosting.predict(X_test)
    gdpg[target_rf][row_count_test_a]=RF.predict(X_test)    
    
    X_gb=gdpg.drop([target,'year','quarter',target_rf], axis=1)
    
      
    y_gb=gdpg[target_gb]
    X_train_gb = X_gb[0:row_count2a]
    X_test_gb = X_gb[row_count_test:row_count2]
    y_train_gb=y_gb[0:row_count2a]
    y_test_gb= y_gb[row_count_test:row_count2]
    
   
    X_rf=gdpg.drop([target,'year','quarter',target_gb], axis=1)
    
       
    y_rf=gdpg[target_rf]
    X_train_rf = X_rf[0:row_count2a]
    X_test_rf = X_rf[row_count_test:row_count2]
    y_train_rf=y_rf[0:row_count2a]
    y_test_rf= y_rf[row_count_test:row_count2]
    
  
    n_samples2 = row_count2a+1
    indices = np.arange(n_samples2)
    test_size = (n_samples2 // n_folds)
    test_starts = range(test_size + n_samples2 % n_folds, n_samples2, test_size)

    tscv2 = [(np.arange(0,k+test_size-2,1),np.arange(k+test_size-2,k+test_size-1,1)) for k in range(test_size + n_samples2 % n_folds, n_samples2, test_size)]
    print(tscv2)
    
    writer['tscv2'].iloc[k]=tscv2
    
    grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=rs), param_grid=parameters_rf,cv=tscv2, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X_train_rf, y_train_rf)
    rf_para=str(grid_search_rf.best_params_)
    max_depth_rf=float(rf_para.split("{'max_depth': ", 1)[1].split(",", 1)[0])

    n_estimators_rf=int(rf_para.split("n_estimators': ", 1)[1].split("}", 1)[0])
    RF=RandomForestRegressor(n_estimators=n_estimators_rf,max_depth=max_depth_rf, random_state=rs)
    RF.fit(X_train_rf, y_train_rf)


    grid_search_gb = GridSearchCV(estimator=GradientBoostingRegressor(random_state=rs), param_grid=parameters_gb, cv=tscv, scoring='neg_mean_squared_error')
    grid_search_gb.fit(X_train_gb, y_train_gb)
    gb_para=str(grid_search_gb.best_params_)

    learning_rate_gb=float(gb_para.split("{'learning_rate': ", 1)[1].split(",", 1)[0])
    n_estimators_gb=int(gb_para.split("n_estimators': ", 1)[1].split("}", 1)[0])
    max_depth_gb=float(gb_para.split("max_depth': ", 1)[1].split(",", 1)[0])

    gradboosting=GradientBoostingRegressor(n_estimators=n_estimators_gb, learning_rate=learning_rate_gb, random_state=rs, max_depth=max_depth_gb)
    gradboosting.fit(X_train_gb, y_train_gb) 


    writer['rf.best_params'].iloc[k]=str(grid_search_rf.best_params_)
    writer['gb.best_params'].iloc[k]=str(grid_search_gb.best_params_)

    writer['year'].iloc[k]=gdpg['year'].iloc[row_count_test]

    writer['RF'].iloc[k]=RF.predict(X_test_rf)

    writer['gradboosting'].iloc[k]=gradboosting.predict(X_test_gb)



writer=writer[0:lastyear-(testyear-1)]

writer['imf_f']=0
writer['imf_s']=0
writer['boj_f']=0
writer['boj_s']=0

for p in range(testyear-startyear, lastyear-startyear+1):
    writer['actual rgdpg'].iloc[p-testyear+startyear]= benchmark['actual gdp'].iloc[p]      
    writer['imf_f'].iloc[p-testyear+startyear] = benchmark['imf_f'].iloc[p]
    writer['imf_s'].iloc[p-testyear+startyear] = benchmark['imf_s'].iloc[p]
    writer['boj_f'].iloc[p-testyear+startyear] = benchmark['boj_f'].iloc[p]
    writer['boj_s'].iloc[p-testyear+startyear] = benchmark['boj_s'].iloc[p]


c=[]
for i in range(0,lastyear-(testyear-1)):
    c.append(i)


rsme=pd.DataFrame(index=c)

rsme['rmse_gradboosting']=0
rsme['rmse_RF']=0
rsme['rmse_imf_f']=0
rsme['rmse_boj_f']=0
rsme['rmse_imf_s']=0
rsme['rmse_boj_s']=0

mape=pd.DataFrame(index=c)

mape['mape_gradboosting']=0
mape['mape_RF']=0
mape['mape_imf_f']=0
mape['mape_boj_f']=0
mape['mape_imf_s']=0
mape['mape_boj_s']=0


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


for i in range(0, lastyear-testyear+1):
    
    r2_RF = r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['RF'].iloc[0:i+1] )
    rsme['rmse_RF'].iloc[i] = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['RF'].iloc[0:i+1] ) )
    mape['mape_RF'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['RF'].iloc[0:i+1] )
    
    r2_gradboosting = r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['gradboosting'].iloc[0:i+1] )
    rsme['rmse_gradboosting'].iloc[i]  = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['gradboosting'].iloc[0:i+1] ) )
    mape['mape_gradboosting'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['gradboosting'].iloc[0:i+1] )
    
    r2_imf_f = r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['imf_f'].iloc[0:i+1] )
    rsme['rmse_imf_f'].iloc[i]  = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['imf_f'].iloc[0:i+1] ) )
    mape['mape_imf_f'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['imf_f'].iloc[0:i+1] )
    
    r2_boj_f= r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['boj_f'].iloc[0:i+1] )
    rsme['rmse_boj_f'].iloc[i] = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['boj_f'].iloc[0:i+1] ) )
    mape['mape_boj_f'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['boj_f'].iloc[0:i+1] )
    
    r2_imf_s = r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['imf_s'].iloc[0:i+1] )
    rsme['rmse_imf_s'].iloc[i]   = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['imf_s'].iloc[0:i+1] ) )
    mape['mape_imf_s'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['imf_s'].iloc[0:i+1] )
    
    r2_boj_s = r2_score( writer['actual rgdpg'].iloc[0:i+1], writer['boj_s'].iloc[0:i+1] )
    rsme['rmse_boj_s'].iloc[i]  = np.sqrt( mean_squared_error( writer['actual rgdpg'].iloc[0:i+1], writer['boj_s'].iloc[0:i+1] ) )
    mape['mape_boj_s'].iloc[i] = mean_absolute_percentage_error( writer['actual rgdpg'].iloc[0:i+1], writer['boj_s'].iloc[0:i+1] )



writer2=pd.DataFrame()

writer2['year']=writer['year'].copy()
writer2['actual rgdpg']=writer['actual rgdpg'].copy()

writer2['GB']=writer['gradboosting'].copy()
writer2['RF']=writer['RF'].copy()

writer2['imf_f']=writer['imf_f'].copy()
writer2['imf_s']=writer['imf_s'].copy()
writer2['boj_f']=writer['boj_f'].copy()
writer2['boj_s']=writer['boj_s'].copy()

writer2['rmse_gradboosting'] = rsme['rmse_gradboosting'].copy()
writer2['rmse_RF'] = rsme['rmse_RF'].copy()

writer2['rmse_imf_f'] = rsme['rmse_imf_f'].copy()
writer2['rmse_imf_s'] = rsme['rmse_imf_s'].copy()
writer2['rmse_boj_f'] = rsme['rmse_boj_f'].copy()
writer2['rmse_boj_s'] = rsme['rmse_boj_s'].copy()

writer2['mape_gradboosting']=mape['mape_gradboosting'].copy()
writer2['mape_RF']=mape['mape_RF'].copy()
writer2['mape_imf_f']=mape['mape_imf_f'].copy()
writer2['mape_boj_f']=mape['mape_boj_f'].copy()
writer2['mape_imf_s']=mape['mape_imf_s'].copy()
writer2['mape_boj_s']=mape['mape_boj_s'].copy()

writer2['rf.best_params']=writer['rf.best_params']
writer2['gb.best_params']=writer['gb.best_params']

writer2['RF_internal_1q_2q_yoy']=writer['RF_internal_1q_2q_yoy']
writer2['GB_internal_1q_2q_yoy']=writer['GB_internal_1q_2q_yoy']
writer2['rf.best_params_internal']=writer['rf.best_params_internal']
writer2['gb.best_params_internal']=writer['gb.best_params_internal']
writer2['randomstate']=rs
writer2['tscv2']=writer['tscv2']


timestr = time.strftime("%Y%m%d%H%M%S")

file = save_path+timestr+'final_reault'+'.xlsx'

writer = pd.ExcelWriter(file, engine='xlsxwriter')

writer2.to_excel(writer,'final_result', index=False)

writer.save()






