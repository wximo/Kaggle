import pandas as pd
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from datetime import datetime,date
import time

import numpy as np

def loadData(path):
    data = pd.read_csv(path)
    return data


if __name__=='__main__':
    path = "/Users/Ximo/Documents/Kaggle/San_Francisco_Crime_Classification/"
    train = path +"train.csv"
    test = path+"test.csv"

    data = loadData(train)
    test = loadData(test)

    data = data.drop(['Descript','Resolution','X','Y'],axis = 1)
    test = test.drop(['X','Y'],axis = 1)
    df = pd.concat([data,test])
    df = df.reset_index()
    df =df.drop('index',axis=1)
    df = df.reindex_axis(data.columns,axis=1)


    format="%Y-%m-%d %H:%M:%S"


    df['hour']=df['Dates'].map(lambda x: time.strptime(x,format).tm_hour)
    df = df.drop('Dates',axis=1)

 #   df=df.drop('AgeuponOutcome',axis=1)
    df['hour']=pd.factorize(df['hour'])[0]
    df['DayOfWeek']=pd.factorize(df['DayOfWeek'])[0]
    df['PdDistrict']=pd.factorize(df['PdDistrict'])[0]
    

    dd=df.pop('Category')
    df.insert(0,'Category',dd)
    
    df['Category']=pd.factorize(df['Category'])[0]
    df=df.drop('Address',axis=1)
    
    known = df[df['Category'] != -1].values
    unknown = df[df['Category'] == -1].values

    Y=known[:,0]
    X=known[:,1:]
    
    clf = RandomForestClassifier(n_estimators=1000).fit(X,Y)
    #clfsvc = SVC().fit(X,Y)

    pred = clf.predict(unknown[:,1:]).astype(int)
    #pred0 = clfsvc.predict(unknown[:,1:]).astype(int)
    ids = range(len(pred))
    
    ids = [i+1 for i in ids]
    print clf.score(X,Y)
    
    pred_prob = clf.predict_proba(unknown[:,1:])
    
    pre_file = open("subm2_p.csv","wb")
    op_f_ob = csv.writer(pre_file)
    #op_f_ob.writerow(['ID','Return_to_owner','Euthanasia','Adoption','Transfer','Died'])
    op_f_ob.writerows(zip(ids,pred_prob))
    pre_file.close()

    
    
