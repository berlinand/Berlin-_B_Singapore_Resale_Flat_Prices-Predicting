import pandas as pd
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import math
from sklearn.model_selection import train_test_split

#This funtion read data and convert into dataframe 
@st.cache_data
def readdata():
   df1= pd.read_csv(r"D:\material\projects\MDE86\project_6_flat_resale\ResaleFlatPricesBasedonApprovalDate19901999.csv")
   df2=pd.read_csv(r"D:\material\projects\MDE86\project_6_flat_resale\ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")
   df3=pd.read_csv(r"D:\material\projects\MDE86\project_6_flat_resale\ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")
   df4=pd.read_csv(r"D:\material\projects\MDE86\project_6_flat_resale\ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")
   df5=pd.read_csv(r"D:\material\projects\MDE86\project_6_flat_resale\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")  
   df6=pd.concat([df1,df2,df3,df4,df5])
   df6= df6.sample(n=50000)
   return df6 

#This function remove unwanted columns
def cleaningdata(df):
   df.drop(['month','remaining_lease','block','street_name'],axis=1,inplace=True)
   return df

#This function remove outliers
def removeoutliers(df):
   cns=['floor_area_sqm','resale_price']
   for cn in cns:
        q1=df[cn].quantile(0.25)
        q3=df[cn].quantile(0.75)

        iqr=q3-q1
        lb=q1-(1.5*iqr)
        ub=q3+(1.5*iqr)

        df=df[(df[cn]>=lb) & (df[cn]<=ub) ]
   return df

#This function add new user data in the dataframe
def adddata(txt,df,clo):
    if len(clo)==1:
       txt1=[(txt)]
       if 'floor_area_sqm' in clo:
             txt1[0]=float(txt1[0])
       elif 'lease_commence_date' in clo:
             txt1[0]=int(txt1[0])
    
    elif len(clo)>1:
          txt1=txt.split(',')
          if 'floor_area_sqm' in clo:
                a=clo.index('floor_area_sqm')
                txt1[a]=float(txt1[a])
          elif 'lease_commence_date' in clo:
                a=clo.index('lease_commence_date')
                txt1[a]=int(txt1[a])


    elif len(clo)==0:
      txt1=txt.split(',')
      txt1[3]=float(txt1[3])
      txt1[5]=int(txt1[5])

    txt1.append(np.nan)
    df.loc[df.index.max() + 1]=txt1
    return df

#This function doing one hot encode to convert catogerical data to numbers
def onehotencoding(df,clo):
    ch=df.select_dtypes(include=['object']).columns.tolist()
    if 'storey_range' in clo:
        ch.remove('storey_range')
    df1=pd.get_dummies(df,columns=ch,dtype=int)
    return df1

#THis  function seperate storey range into two different columns
def storyrange(df):
      df[['storey_start', 'storey_end']] = df['storey_range'].str.split('TO',expand=True)
      df['storey_start'] = pd.to_numeric(df['storey_start'])
      df['storey_end'] = pd.to_numeric(df['storey_end'])
      df.drop(['storey_range'],axis=1,inplace=True)
      
      return df 
  
#This function remove outliers
def remout(df,clo):
   cns=['storey_start','storey_end','lease_commence_date']
   if len(clo)!=0:
         if 'storey_range' in clo  and cns[2] in clo:
             cns=cns
         elif 'storey_range' in clo:
               cns=[cns[0],cns[1]]
         
         elif cns[2] in clo:
               cns=[cns[2]]
       
   
   for cn in cns:
        q1=df[cn].quantile(0.25)
        q3=df[cn].quantile(0.75)

        iqr=q3-q1
        lb=q1-(1.5*iqr)
        ub=q3+(1.5*iqr)

        df=df[(df[cn]>=lb) & (df[cn]<=ub) ]
   return df

#This function predict the resale price using ml models 
def mmodels(df,lo,models,test_si,modelrun,ll1,ll2,ll3,ll4,ll5,ll6,ll7,ll8,ll9):
  for i  in range(modelrun):
   X=df.drop(['resale_price'],axis=1)
   y=df['resale_price']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_si,random_state=i,shuffle=True)

 
   if models=='Linear Regression':

               model=LinearRegression(fit_intercept=ll1,copy_X=ll2,n_jobs=ll3,positive=ll4)
   if models=='Lasso':
               model=Lasso(alpha=ll1,fit_intercept=ll2,precompute=ll3,copy_X=ll4,max_iter=ll5,
                           warm_start=ll6,positive=ll7,tol=ll8,selection=ll9)
   if models=='Ridge':
               model=Ridge(alpha=ll1,fit_intercept=ll2,copy_X=ll3,max_iter=ll4,tol=ll5,positive=ll6,solver=ll7)        
   if models=='Decision Tree Regression':
               model=DecisionTreeRegressor(criterion=ll1,splitter=ll2,max_depth=ll3,min_samples_split=ll4,
                                           min_samples_leaf=115,min_weight_fraction_leaf=ll6)
                  
   if models=='Random Forest Regression':
               model=RandomForestRegressor(n_estimators=ll1,criterion=ll2,max_depth=ll3,min_samples_split=ll4,
                                           min_samples_leaf=ll5,min_weight_fraction_leaf=ll6) 
   if models=='Gradient Boosting Regression':
               model=GradientBoostingRegressor(loss=ll1,learning_rate=ll2,n_estimators=ll3,subsample=ll4,
                                               criterion=ll5,min_samples_split=ll6,min_samples_leaf=ll7,
                                               min_weight_fraction_leaf=ll8,max_depth=ll9)  
   model.fit(X_train,y_train)       
   train_pred=model.predict(X_train)
   test_pred=model.predict(X_test)
   test_pred_1=model.predict(lo)
   st.write(F":violet[Train MSE]={mean_squared_error(y_train,train_pred)} ")
   st.write(F":violet[Train RMSE]={math.sqrt(mean_squared_error(y_train,train_pred))} ")
   st.write(F":violet[Train MAE]={mean_absolute_error(y_train,train_pred)} ")
   st.write(F":violet[Train r2 score]={r2_score(y_train,train_pred)} ")
   
   st.write(F":violet[Test MSE]={mean_squared_error(y_test,test_pred)} ")
   st.write(F":violet[Test RMSE]={math.sqrt(mean_squared_error(y_test,test_pred))} ")
   st.write(F":violet[Test MAE]={mean_absolute_error(y_test,test_pred)} ")
   st.write(F":violet[Test r2 score]={r2_score(y_test,test_pred)} ")
   
   r2=r2_score(y_test,test_pred)
  
   st.write(f':blue[selling price]= {''.join(test_pred_1.astype(str))}')
   st.write("-----------------------------------------------------")   
try:
  st.title(":blue[Singapore Flat] :orange[ Prices]:blue[ Predict]")
  df=readdata()
  df=cleaningdata(df)
  cl=list(df.columns)
  cl.remove("resale_price")
  oba=st.selectbox(label='OUTLIERS',options=['Before','After'])
  clo=st.multiselect(label="select column",options=cl)
  if oba=='After':
       df=removeoutliers(df)
  txt=st.text_input(label='town,flat type,storey range,floor area in sqm,flat model,lease commence date')
  co1,co2,co3=st.columns(3)

  models=co1.selectbox(label='models',options=['Linear Regression','Lasso','Ridge','Decision Tree Regression',
                                                          'Random Forest Regression', 'Gradient Boosting Regression'])
  test_si=co2.selectbox(label="test data ",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

  modelrun=co3.number_input(label="run model",min_value=1,max_value=100)
  if len(txt)>0:
      if len(clo)!=0:
            df=df.copy()
            clo.append("resale_price")
            df=df[clo]
      
      df=adddata(txt,df,clo) 
      df=onehotencoding(df,clo)
      if 'storey_range' in clo:
           df=storyrange(df)
      
      lo=df.copy()
      lo=lo.tail(1)
      if 'storey_range' in clo or 'lease_commence_date' in clo:
                   df=remout(df,clo)
      lo.drop(['resale_price'],axis=1,inplace=True)
      df=df.copy()
      df = df.drop(df.index[-1])
      
      if models=='Linear Regression':
        l1,l2,l3,l4=st.columns(4)
        ll1=l1.selectbox(label='fit_intercept',options=[True,False])
        ll2=l2.selectbox(label='copy_X',options=[True,False])
        ll3=l3.selectbox(label='n_jobs',options=[1,-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        ll4=l4.selectbox(label='positive',options=[True,False])
        ll5='nope'
        ll6='nope'
        ll7='nope'
        ll8='nope'
        ll9='nope'
      if models=='Lasso':
         l1,l2,l3,l4,l5=st.columns(5) 
         l6,l7,l8,l9=st.columns(4)
         ll1=l1.selectbox(label='alpha',options=[float(x) for x in range(1,20) ])
         ll2=l2.selectbox(label='fit_intercept',options=[True,False])
         ll3=l3.selectbox(label='precompute',options=[False,True])
         ll4=l4.selectbox(label='copy_X',options=[True,False])
         ll5=l5.selectbox(label='max_iter',options=[x for x in range(100,10000,100)])
         ll6=l6.selectbox(label='warm_start',options=[False,True])
         ll7=l7.selectbox(label='positive',options=[True,False])
         ll8=l8.selectbox(label='tol',options=[0.0001,0.0002,0.0003,
                                               0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,
                                               0.002,0.003,0.004,0.005,0.006,0.007,0.009,0.01])
         ll9=l9.selectbox(label='selection',options=['cyclic','random'])

      if models=='Ridge':
         l1,l2,l3,l4=st.columns(4) 
         l5,l6,l7=st.columns(3)
         ll1=l1.selectbox(label='alpha',options=[float(x) for x in range(1,20) ])  
         ll2=l2.selectbox(label='fit_intercept',options=[True,False])
         ll3=l3.selectbox(label='copy_X',options=[True,False])
         ll4=l4.selectbox(label='max_iter',options=[x for x in range(100,100000,100)])
         ll5=l5.selectbox(label='tol',options=[0.0001,0.0002,0.0003,
                                               0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,
                                               0.002,0.003,0.004,0.005,0.006,0.007,0.009,0.01])
         ll6=l6.selectbox(label='positive',options=[True,False])
         ll7=l7.selectbox(label="solver",options=['auto','svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'])
         ll8='nope'
         ll9='nope'

      if models=='Decision Tree Regression':
            l1,l2,l3,l4,l5,l6=st.columns(6) 
            ll1=l1.selectbox(label='criterion',options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
            ll2=l2.selectbox(label='splitter',options=['best', 'random'])
            ne=[x for x in range(1,10000)]
            ne.insert(0,None)
            ll3=l3.selectbox(label='max_depth',options=ne)
            ll4=l4.selectbox(label='min_samples_split',options=[x for x in range(2,10000)])
            ll5=l5.selectbox(label='min_samples_leaf',options=[x for x in range(1,10000)])
            ll6=l6.selectbox(label='min_weight_fraction_leaf',options=[0.0,0.1,0.2,0.3,0.4,0.5])
            ll7='nope'
            ll8='nope'
            ll9='nope'
      if models=='Random Forest Regression':
           l1,l2,l3,l4,l5,l6=st.columns(6) 
           ll1=l1.selectbox(label='n_estimators',options=[x for x in range(100,1000)])
           ll2=l2.selectbox(label='criterion',options=['squared_error', 'friedman_mse', 'absolute_error', 'poisson'])
           ne=[x for x in range(1,10000)]
           ne.insert(0,None)
           ll3=l3.selectbox(label='max_depth',options=ne)
           ll4=l4.selectbox(label='min_samples_split',options=[x for x in range(2,10000)])
           ll5=l5.selectbox(label='min_samples_leaf',options=[x for x in range(1,10000)])  
           ll6=l6.selectbox(label='min_weight_fraction_leaf',options=[0.0,0.1,0.2,0.3,0.4,0.5])
           ll7='nope'
           ll8='nope'
           ll9='nope'
      if models=='Gradient Boosting Regression':
           l1,l2,l3,l4,l5=st.columns(5) 
           l6,l7,l8,l9=st.columns(4)
           ll1=l1.selectbox(label='loss',options=['squared_error', 'absolute_error', 'huber', 'quantile'])
           ll2=l2.selectbox(label='learning_rate',options=[float(x)/10 for x in range(1,1000) ]) 
           ll3=l3.selectbox(label='n_estimators',options=[x for x in range(100,10000)]) 
           ll4=l4.selectbox(label='subsample',options=[float(x)/10 for x in range(1,11)])
           ll5=l5.selectbox(label='criterion',options=['friedman_mse', 'squared_error'])
           ll6=l6.selectbox(label='min_samples_split',options=[x for x in range(2,10000)])
           ll7=l7.selectbox(label='min_samples_leaf',options=[x for x in range(1,10000)])  
           ll8=l8.selectbox(label='min_weight_fraction_leaf',options=[0.0,0.1,0.2,0.3,0.4,0.5])      
           ne=[x for x in range(1,10000)]
           ne.insert(0,None)
           ll9=l9.selectbox(label='max_depth',options=ne)
      mmodels(df,lo,models,test_si,modelrun,ll1,ll2,ll3,ll4,ll5,ll6,ll7,ll8,ll9)

  
except Exception as e:
  
  if str(e).strip() =='Maximum number of iterations reached.':
    st.write(":blue[fit intercept can't find a solution]")    
  else:
    st.warning(f":red[{e}]")
  
