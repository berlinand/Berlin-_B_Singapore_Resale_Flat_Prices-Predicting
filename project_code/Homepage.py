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
def adddata(txt,df):
    txt1=txt.split(',')
    txt1[3]=float(txt1[3])
    txt1[5]=int(txt1[5])
    txt1.insert(6,np.nan)
    df.loc[df.index.max() + 1]=txt1
    return df

#This function doing one hot encode to convert catogerical data to numbers
def onehotencoding(df):
    df1=pd.get_dummies(df,columns=['town','flat_type','flat_model'],dtype=int)
    return df1

#THis  function seperate storey range into two different columns
def storyrange(df):
      df[['storey_start', 'storey_end']] = df['storey_range'].str.split('TO',expand=True)
      df['storey_start'] = pd.to_numeric(df['storey_start'])
      df['storey_end'] = pd.to_numeric(df['storey_end'])
      df.drop(['storey_range'],axis=1,inplace=True)
      return df 
  
#This function remove outliers
def remout(df):
   cns=['storey_start','storey_end','lease_commence_date']
   for cn in cns:
        q1=df[cn].quantile(0.25)
        q3=df[cn].quantile(0.75)

        iqr=q3-q1
        lb=q1-(1.5*iqr)
        ub=q3+(1.5*iqr)

        df=df[(df[cn]>=lb) & (df[cn]<=ub) ]
   return df

#This function predict the resale price using ml models 
def mmodels(df,lo,models,test_si,modelrun):
  for i  in range(modelrun):
   X=df.drop(['resale_price'],axis=1)
   y=df['resale_price']
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_si,random_state=i,shuffle=True)

 
   if models=='Linear Regression':
               model=LinearRegression()
   if models=='Lasso':
               model=Lasso(alpha=1.0,random_state=i,max_iter=1000)
   if models=='Ridge':
               model=Ridge()        
   if models=='Decision Tree Regression':
               model=DecisionTreeRegressor(max_depth=i+30)
                  
   if models=='Random Forest Regression':
               model=RandomForestRegressor(max_depth=i+30) 
   if models=='Gradient Boosting Regression':
               model=GradientBoostingRegressor(max_depth=i+30)  
   model.fit(X_train,y_train)       
   train_pred=model.predict(X_train)
   test_pred=model.predict(X_test)
   test_pred_1=model.predict(lo)
   st.write(F":violet[Train MSE]-{mean_squared_error(y_train,train_pred)} ")
   st.write(F":violet[Train RMSE]-{math.sqrt(mean_squared_error(y_train,train_pred))} ")
   st.write(F":violet[Train MAE]-{mean_absolute_error(y_train,train_pred)} ")
   st.write(F":violet[Train r2 score]-{r2_score(y_train,train_pred)} ")
   
   st.write(F":violet[Test MSE]-{mean_squared_error(y_test,test_pred)} ")
   st.write(F":violet[Test RMSE]-{math.sqrt(mean_squared_error(y_test,test_pred))} ")
   st.write(F":violet[Test MAE]-{mean_absolute_error(y_test,test_pred)} ")
   st.write(F":violet[Test r2 score]-{r2_score(y_test,test_pred)} ")
   
   r2=r2_score(y_test,test_pred)
   tx=''.join(test_pred_1.astype(str))
   st.write(f':blue[selling price]= {tx}')
   st.write("-----------------------------------------------------")   
try:
  df=readdata()
  df=cleaningdata(df)
  oba=st.selectbox(label='OUTLIERS',options=['Before','After'])
  if oba=='After':
       df=removeoutliers(df)
  txt=st.text_input(label='town,flat type,storey range,floor area in sqm,flat model,lease commence date')
  co1,co2,co3=st.columns(3)

  models=co1.selectbox(label='models',options=['Linear Regression','Lasso','Ridge','Decision Tree Regression',
                                                          'Random Forest Regression', 'Gradient Boosting Regression'])
  test_si=co2.selectbox(label="test data ",options=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

  modelrun=co3.number_input(label="Train model",min_value=1,max_value=100)
  if len(txt)>0:
  
      df=adddata(txt,df)
      df=onehotencoding(df)
      df=storyrange(df)
      lo=df.copy()
      lo=lo.tail(1)
      df=remout(df)
      lo.drop(['resale_price'],axis=1,inplace=True)
      df=df.copy()
      df = df.drop(df.index[-1])
      mmodels(df,lo,models,test_si,modelrun)
  
except Exception as e:
      
  st.warning(f":red[{e}]")
