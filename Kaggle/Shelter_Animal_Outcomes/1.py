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
    path = "/Users/Ximo/Documents/Kaggle/Shelter_Animal_Outcomes/"
    train = path +"train.csv"
    test = path+"test.csv"

    data = loadData(train)
    test = loadData(test)

    data = data.drop(['AnimalID','OutcomeSubtype'],axis = 1)
    df = pd.concat([data,test])
    df = df.reset_index()
    df =df.drop('index',axis=1)
    df = df.reindex_axis(data.columns,axis=1)

    #df=df.drop('Name',axis=1)

    df['Name'][df['Name'].isnull()]='no one'

    format="%Y-%m-%d %H:%M:%S"
    t1=time.strptime("2008-01-31 00:11:23",format)

    #df['DateTime']=df['DateTime'].map(lambda x: int(x.split(' ')[1].split(':')[0]))
    df['week']=df['DateTime'].map(lambda x: time.strptime(x,format).tm_wday)
    df['hour']=df['DateTime'].map(lambda x: time.strptime(x,format).tm_hour)
    df = df.drop('DateTime',axis=1)
    #df['SexuponOutcome'][df['SexuponOutcome'].isnull()]=df['SexuponOutcome'].mode().values
    
    #df['AgeuponOutcome'][df['AgeuponOutcome'].isnull()]=df['AgeuponOutcome'].mode().values


    def age_days(age):
        a=age.split(' ')
        b=0
        if a[-1] == "years" or a[-1] == "year":
            b+=int(a[0])*365
        elif a[-1]=="month" or a[-1]=="monthes":
            b+=int(a[0])*30
        elif a[-1]=="week" or a[-1]=="weeks":
            b+=int(a[0])*7
        else:
            b+=int(a[0])
        return b

    #df['AgeuponOutcome']=df['AgeuponOutcome'].map(lambda x: age_days(x))
    df['AgeuponOutcome'][df['AgeuponOutcome'].notnull()]=df['AgeuponOutcome'][df['AgeuponOutcome'].notnull()].map(lambda x: age_days(x))
    df['AgeuponOutcome']=pd.factorize(df['AgeuponOutcome'])[0]    
    df['AgeuponOutcome'][df['AgeuponOutcome'].isnull()]=df['AgeuponOutcome'][df['AgeuponOutcome'] != -1].median()

    df['SexuponOutcome']=pd.factorize(df['SexuponOutcome'])[0]    
    df['SexuponOutcome'][df['SexuponOutcome'].isnull()]=df['SexuponOutcome'][df['SexuponOutcome'] != -1].median()



 #   df=df.drop('AgeuponOutcome',axis=1)
    #df['SexuponOutcome']=pd.factorize(df['SexuponOutcome'])[0]
    df['AnimalType']=pd.factorize(df['AnimalType'])[0]
    df['Name']=pd.factorize(df['Name'])[0]
    #df=df.drop('Name',axis=1)
    #df['DateTime']=pd.factorize(df['DateTime'])[0]
    #df['AgeuponOutcome']=pd.factorize(df['AgeuponOutcome'])[0]
    df['Color']=pd.factorize(df['Color'])[0]
    df['Breed']=pd.factorize(df['Breed'])[0]
    df['OutcomeType']=pd.factorize(df['OutcomeType'])[0]

    dd=df.pop('OutcomeType')
    df.insert(0,'OutcomeType',dd)
    
    known = df[df['OutcomeType'] != -1].values
    unknown = df[df['OutcomeType'] == -1].values

    Y=known[:,0]
    X=known[:,1:]
    
    clf = RandomForestClassifier(n_estimators=2500).fit(X,Y)
    #clf = SVC(probability = True).fit(X,Y)

    
    #pred0 = clfsvc.predict(unknown[:,1:]).astype(int)
    
    
    pred_prob = clf.predict_proba(unknown[:,1:])
    ids = range(len(pred_prob))
    ids = [i+1 for i in ids]
    
    print clf.score(X,Y)
    
    pre_file = open("subm3_p.csv","wb")
    op_f_ob = csv.writer(pre_file)
    op_f_ob.writerow(['ID','Return_to_owner','Euthanasia','Adoption','Transfer','Died'])
    op_f_ob.writerows(zip(ids,pred_prob[:,0],pred_prob[:,1],pred_prob[:,2],pred_prob[:,3],pred_prob[:,4]))
    pre_file.close()

    
    
