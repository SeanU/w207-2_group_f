# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:08:33 2016

@author: Subhashini
"""


import pandas as pd
import numpy as np
from datetime import date, datetime
# SK-learn libraries for feature extraction from text.
from sklearn.feature_extraction.text import *

from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier

# SK-learn libraries for evaluation.
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import cross_validation

def age_years(x):
    if(pd.isnull(x)):
        return 0
    
    age = int(x.split(' ')[0])
        
    if 'month' in x:
        age /= 12.0

    elif 'week' in x:
        age /= 52.0

    elif 'day' in x:
        age /= 365.0

    elif 'year' in x:
        age = age
    else:
        age = 0

    return age

'''
#exploring breed names
[s for s in train['Breed'] if ('/' in s)  & ('mix' in s.lower())]
#Jack Russell Terrier/Unknown - 1
[s for s in train['Breed'] if ('/' in s)  & ('known' in s.lower())]
#unknown mix
[s for s in train['Breed'] if ('/' not in s)  & ('known' in s.lower())]
#<breed> mix
[s for s in train['Breed'] if ('/' not in s)  & ('mix' in s.lower())]
'''

def get_breed(line):
    #special case this miscoded value : should be Black and Tan hound Mix (Black and Tan hound is a breed)
    if line == 'Black/Tan Hound Mix':
        return 'purebreed_mix'
        
    if '/' in line:
        breed2 = line.split('/')[1]
        if 'unknown' in breed2:
            return 'purebreed_unknown'
        else:
            return 'purebreed_purebreed'
    elif 'unknown' in line:
        return "unknown"
    elif 'mix' in line:
      return "purebreed_mix"
    else:
        return "purebreed"


def get_color(x):
    if pd.isnull(x):
        return x
        
    colors = x.strip().split('/')
    if len(colors) == 1:
        return "single_color"
    elif len(colors) == 2:
        return "double_color"
    else:
        return "twoplus_color" 

#---------------------------------------------------------------------------------------------------

train = pd.read_csv('C:\\Subha\\WS207-ML\\Kaggle\\git\\train.csv', parse_dates=['DateTime'])
test = pd.read_csv('C:\\Subha\\WS207-ML\\Kaggle\\git\\test.csv', parse_dates=['DateTime'])

# Exploratory data analysis
train.describe()

# We are going to modify all the data test+train together to generate our new features then break them back up for simplicity sake.

# get number of training elements
train_num = train.shape[0]
#df.rename(columns={'$a': 'a', '$b': 'b'}, inplace=True)
train.rename(columns={'AnimalID': 'ID'}, inplace=True)
all_data = pd.concat((train, test), axis=0, ignore_index=True)

print train.shape     #26,729 rows
print test.shape      #11,456
print all_data.shape  #38,185

all_data.describe()

#-------------------------------------------------------------------------------------------------

#clean up and recode data


all_data['OutcomeType']= all_data['OutcomeType'].str.lower()
all_data['OutcomeType']= all_data['OutcomeType'].str.strip()
all_data['OutcomeType']= all_data['OutcomeType'].astype('category')
#all_data['OutcomeType2'] = pd.Categorical.from_array(all_data.OutcomeType).codes
#print all_data['OutcomeType2'].value_counts() #-1 is the NaN for test data
# all the test data as expected does not have an outcometype
print sum(pd.isnull(all_data['OutcomeType'])) 
#Nans were coded to -1 in the numerical data
#print sum(pd.isnull(all_data['OutcomeType2'])) 
#print sum(all_data['OutcomeType2'] == -1) 

#--------------------------------------------------------------------------------------------------

#does it have a name?
all_data['Name'] = all_data['Name'].str.strip()
all_data['Hasname'] = all_data['Name'].map(lambda x: 1 if not pd.isnull(x) and x else 0)
all_data['Hasname'].value_counts()

#-------------------------------------------------------------------------------------------------'

all_data['AnimalType'] = all_data['AnimalType'].str.lower()
all_data['AnimalType'] = all_data['AnimalType'].str.strip()
all_data['Iscat'] = all_data['AnimalType'].map(lambda x: 1 if x=='cat' else 0)
#no missing values
print sum(pd.isnull(all_data['Iscat']))  

#--------------------------------------------------------------------------------------------------

all_data['SexuponOutcome'] = all_data['SexuponOutcome'].str.lower()
all_data['SexuponOutcome'] = all_data['SexuponOutcome'].str.strip()
all_data['SexuponOutcome'] = all_data['SexuponOutcome'].astype('category')
print all_data['SexuponOutcome'].value_counts()
#1 missing value, convert it to Unknown category
print sum(pd.isnull(all_data['SexuponOutcome']))
all_data['SexuponOutcome'][pd.isnull(all_data['SexuponOutcome'])] = "unknown"

all_data = all_data.join(pd.get_dummies(all_data['SexuponOutcome']))

#--------------------------------------------------------------------------------------------------

all_data['AgeuponOutcome'] = all_data['AgeuponOutcome'].str.lower()
all_data['AgeuponOutcome'] = all_data['AgeuponOutcome'].str.strip()
all_data['Ageyears'] = map(age_years, all_data['AgeuponOutcome'])
all_data['Ageyears'].describe()
#24 values are unknkown
print sum(pd.isnull(all_data['Ageyears'])) 


#cat and dog age categories 
#source - http://www.akc.org/learn/family-dog/how-to-calculate-dog-years-to-human-years/
#http://icatcare.org/advice/how-guides/how-tell-your-cat%E2%80%99s-age-human-years
age_labels_dogs=['baby', 'adolescent', 'adult', 'senior']
age_ranges_dogs=[0.0, 1.0, 2.0, 7.0, 30.0]
age_ranges_cats=[0.0, 0.5, 2.0, 6.0, 10.0,15.0, 30.0]
age_labels_cats=['baby', 'adolescent', 'adult','mature', 'senior', 'geriatric']


#compute the dog age and cat age categories separately, then merge them
all_data['dog_ages'] = pd.cut(all_data['Ageyears'][all_data['AnimalType']=='dog'], age_ranges_dogs,  labels=age_labels_dogs)
all_data['cat_ages'] = pd.cut(all_data['Ageyears'][all_data['AnimalType']=='cat'], age_ranges_cats,  labels=age_labels_cats)

all_data['Agecategory'] = [all_data['dog_ages'][x] if not pd.isnull(all_data['dog_ages'][x]) else all_data['cat_ages'][x] for x in range(all_data['dog_ages'].size)]

all_data['Agecategory'] = all_data['Agecategory'].astype('category')
all_data['Agecategory'].value_counts()

#drop the temp columns
all_data.drop(['dog_ages', 'cat_ages'], axis=1, inplace=True)

#the AgeuponOutcome had 24 missing values, whereas the Agecategory has 59. Why? See below.
sum(pd.isnull(all_data['Agecategory']))
# The extra nulls are due to age=0
all_data['AgeuponOutcome'][pd.isnull(all_data['Agecategory'])]


all_data = all_data.join(pd.get_dummies(all_data['Agecategory']))

#--------------------------------------------------------------------------------------------------

all_data['Breed'] = all_data['Breed'].str.lower()
all_data['Breed'] = all_data['Breed'].str.strip()

#breed type
all_data['Breedtype'] = all_data['Breed'].map(get_breed)
all_data['Breedtype'].value_counts()

all_data = all_data.join(pd.get_dummies(all_data['Breedtype'], prefix='breed'))

all_data['Color'] = all_data['Color'].str.lower()
all_data['Color'] = all_data['Color'].str.strip()

all_data['Colortype'] = all_data['Color'].map(get_color)
all_data['Colortype'].value_counts()

#no missing values
sum(pd.isnull(all_data['Color']))
sum(pd.isnull(all_data['Colortype']))


all_data = all_data.join(pd.get_dummies(all_data['Colortype'], prefix='col'))


#---------------------------------------------------------------
all_data.dtypes

all_data['Month'] = all_data.DateTime.apply(lambda d: d.strftime('%B'))   # 'January', 'Feburary', . . .
#all_data['Week'] = all_data.DateTime.apply(lambda d: d.strftime('Day%d'))  # 'Day01', 'Day02', . . .
all_data['WeekDay'] = all_data.DateTime.apply(lambda d: d.strftime('%A')) # 'Sunday', 'Monday', . . .  


all_data.head()

columns_to_code = ['Month', 'WeekDay']

for column in columns_to_code:
    dummies = pd.get_dummies(all_data[column])
    all_data = pd.concat((all_data, dummies), axis=1)



#---------------------------------------------------------------------------

vec = CountVectorizer(min_df=1)
breeds = vec.fit_transform(all_data['Breed'])
breeds = breeds.toarray()

#create new features from breed names
for ii in range(breeds.shape[1]):
    colname = 'breed_%d' %ii
    all_data[colname] = pd.Series(breeds[:,ii])

#--------------------------------------------------------------------------------------------------


vec = CountVectorizer(min_df=1)
colors = vec.fit_transform(all_data['Color'])
colors = colors.toarray()

#create new features from breed names
for ii in range(colors.shape[1]):
    colname = 'color_%d' %ii
    all_data[colname] = pd.Series(colors[:,ii])



#--------------------------------------------------------------------------------------------------


[item for item in all_data.columns]

#--------------------------------------------------------------------------------------------------



# Lets break things back up into our test and train data sets.
train_data = all_data.iloc[:train_num]
test_data = all_data.iloc[train_num:]
test_ids = all_data['ID'][train_num:].values
train_labels = train_data.OutcomeType

train_data.shape, test_data.shape

#drop the columns we don't need
cols_to_drop = [
 'AgeuponOutcome',
 'AnimalType',
 'Breed',
 'Color',
 'DateTime',
 'Name',
 'OutcomeSubtype',
 'SexuponOutcome',
 'Colortype',
 'Breedtype',
 'Agecategory',
 'Month', 'WeekDay',
 'ID', 'OutcomeType']

train_data = train_data.drop(cols_to_drop, axis=1)
test_data = test_data.drop(cols_to_drop, axis=1)

train_data.head()

[item for item in train_data.columns]

#----------------------------------------------------------------------------

#now we're ready to fit models!


#----------------------------------------------------------------------------
#multinomial naive bayes

#with categorical, with binary,with date

# Get the probability predictions for computing the log-loss function
kf = KFold(train_data.shape[0], n_folds=6)
# prediction probabilities number of samples, by number of classes
y_pred = np.zeros((len(train_labels),len(set(train_labels))))
for train_index, test_index in kf:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = train_data.iloc[train_index], train_data.iloc[test_index]
    y_train, y_test = train_labels.iloc[train_index], train_labels.iloc[test_index]
    print X_train.shape, X_test.shape
    #clf = BernoulliNB() #1.44
    #clf = MultinomialNB() #1.29, 1.21
    #clf = RandomForestClassifier(n_estimators=100, n_jobs=3) #1.75, 1.69
    #clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=1000) #1.6, 1.6
    #clf = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)    #8.6
    clf = LogisticRegression() #0.897, 0.88
    #clf = LogisticRegression(penalty="l1") #0.89, 0.839with ALL date features
    #use GridSearchCV to find the optimum value for K (# of neighbors)
    #neighbors = {'n_neighbors': range(1,30)}    
    #knn = KNeighborsClassifier(n_neighbors=200) #1.44 for all values!
    #clf = GridSearchCV(knn, neighbors)
    clf.fit(X_train, y_train)
    y_pred[test_index] = clf.predict_proba(X_test)
    
metrics.log_loss(train_labels, y_pred)    

#------------Submission-------------------

#train on the entire training set with the chosen model
clf = LogisticRegression()
y_pred_sub = clf.fit(train_data, train_labels).predict_proba(test_data)


# Prepare the submission file
sub = pd.DataFrame(y_pred_sub, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
sub.insert(0, 'ID', test_ids.astype(int))

sub.head()

sub.to_csv("C:\\Subha\\WS207-ML\\Kaggle\\git\\submission.csv", index=False)

#----------------------------------- GMM

#cats and dogs by outcome type
train_groups = all_data[:train_num].groupby(['AnimalType', 'OutcomeType'])
for name, group in train_groups:
    print name, "\t\t", len(group)

sum(all_data['AnimalType'][:train_num] == 'dog') #15595
sum(all_data['AnimalType'][:train_num] == 'cat') #11134






