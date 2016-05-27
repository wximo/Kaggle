# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 09:36:40 2016

@author: Administrator
"""
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
from sklearn import svm

def parse_date(date):
    tem = date.split()
    datetime, time = tem[0], tem[1]
    tem = datetime.split('-')
    year, month, day = tem[0], tem[1], tem[2]
    tem = time.split(':')
    hour = tem[0]
    return year, month, day, hour

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Get binarized weekdays, districts, and parse Dates to year, month, day, hour
weekdays = pd.get_dummies(train.DayOfWeek)
district = pd.get_dummies(train.PdDistrict)
parse_res = map(parse_date, train.Dates.values)
years = pd.Series(data=map(lambda x: x[0], parse_res), name='year')
months = pd.Series(data=map(lambda x: x[1], parse_res), name='month')
days = pd.Series(data=map(lambda x: x[2], parse_res), name='day')
hours = pd.Series(data=map(lambda x: x[3], parse_res), name='hour')
categorys = train.Category

# Build new array
train_data = pd.concat([categorys, hours, days, months, years, weekdays, district], axis=1)

# Repeat for test data
weekdays = pd.get_dummies(test.DayOfWeek)
district = pd.get_dummies(test.PdDistrict)
parse_res = map(parse_date, test.Dates.values)
years = pd.Series(data=map(lambda x: x[0], parse_res), name='year')
months = pd.Series(data=map(lambda x: x[1], parse_res), name='month')
days = pd.Series(data=map(lambda x: x[2], parse_res), name='day')
hours = pd.Series(data=map(lambda x: x[3], parse_res), name='hour')

test_data = pd.concat([hours, days, months, years, weekdays, district], axis=1)

mnb = MultinomialNB()
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
            'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
            'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
train_x = pd.DataFrame(train_data, columns=features).values
train_y = train_data.Category.values
mnb.fit(train_x, train_y)
pred = mnb.predict_proba(train_x)
print log_loss(train_y, pred)
print mnb.score(train_x,train_y)

test_x = pd.DataFrame(test_data, columns=features).values
predicted = mnb.predict_proba(test_x)
# Write results
result = pd.DataFrame(predicted, columns=mnb.classes_)
result.to_csv('testResult.csv', index = True, index_label = 'Id' )
    