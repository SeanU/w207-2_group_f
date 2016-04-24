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


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from matplotlib.colors import LogNorm
from scipy.spatial.distance import cdist


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
print sum(pd.isnull(all_data['OutcomeType'])) 

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

all_data = pd.concat((all_data,pd.get_dummies(all_data['SexuponOutcome'])), axis=1)

#--------------------------------------------------------------------------------------------------

all_data['AgeuponOutcome'] = all_data['AgeuponOutcome'].str.lower()
all_data['AgeuponOutcome'] = all_data['AgeuponOutcome'].str.strip()
all_data['Ageyears'] = map(age_years, all_data['AgeuponOutcome'])
all_data['Ageyears'].describe()
#24 values are unknkown
print sum(pd.isnull(all_data['Ageyears'])) 


#create cat and dog age categories 
#source - http://www.akc.org/learn/family-dog/how-to-calculate-dog-years-to-human-years/
#http://icatcare.org/advice/how-guides/how-tell-your-cat%E2%80%99s-age-human-years
age_labels_dogs=['baby', 'adolescent', 'adult', 'senior']
age_ranges_dogs=[0.0, 1.0, 2.0, 7.0, 30.0]
age_ranges_cats=[0.0, 0.5, 2.0, 6.0, 10.0,15.0, 30.0]
age_labels_cats=['baby', 'adolescent', 'adult','mature', 'senior', 'geriatric']


#compute the dog age and cat age categories separately, then merge them
all_data['dog_ages'] = pd.cut(all_data['Ageyears'][all_data['AnimalType']=='dog'], age_ranges_dogs,  labels=age_labels_dogs)
all_data['cat_ages'] = pd.cut(all_data['Ageyears'][all_data['AnimalType']=='cat'], age_ranges_cats,  labels=age_labels_cats)
#merge
all_data['Agecategory'] = [all_data['dog_ages'][x] if not pd.isnull(all_data['dog_ages'][x]) else all_data['cat_ages'][x] for x in range(all_data['dog_ages'].size)]

all_data['Agecategory'] = all_data['Agecategory'].astype('category')
all_data['Agecategory'].value_counts()

#drop the temp columns
all_data.drop(['dog_ages', 'cat_ages'], axis=1, inplace=True)

#the AgeuponOutcome had 24 missing values, whereas the Agecategory has 59. Why? See below. # The extra nulls are due to age=0
sum(pd.isnull(all_data['Agecategory']))
all_data['AgeuponOutcome'][pd.isnull(all_data['Agecategory'])]

all_data = pd.concat((all_data, pd.get_dummies(all_data['Agecategory'])), axis=1)
#--------------------------------------------------------------------------------------------------

all_data['Breed'] = all_data['Breed'].str.lower()
all_data['Breed'] = all_data['Breed'].str.strip()

#breed type
all_data['Breedtype'] = all_data['Breed'].map(get_breed)
all_data['Breedtype'].value_counts()

all_data = pd.concat((all_data, pd.get_dummies(all_data['Breedtype'])), axis=1)

all_data['Color'] = all_data['Color'].str.lower()
all_data['Color'] = all_data['Color'].str.strip()

all_data['Colortype'] = all_data['Color'].map(get_color)
all_data['Colortype'].value_counts()

#no missing values
sum(pd.isnull(all_data['Color']))
sum(pd.isnull(all_data['Colortype']))

all_data = pd.concat((all_data, pd.get_dummies(all_data['Colortype'], prefix='col')), axis=1)


#---------------------------------------------------------------
all_data.dtypes

all_data['Hour'] = all_data.DateTime.apply(lambda d: d.strftime('%H')).astype('int') # [00,23]
all_data['Ampm'] = all_data.DateTime.apply(lambda d: d.strftime('%p')) # Am/pm
all_data['Dayofweek'] = all_data.DateTime.apply(lambda d: d.strftime('%w')).astype('int') # [0(sunday),6]
all_data['Dayofmonth'] = all_data.DateTime.apply(lambda d: d.strftime('%d')).astype('int') # 01,02,..31
all_data['Month'] = all_data.DateTime.apply(lambda d: d.strftime('%m')).astype('int')   # Month as a number
all_data['Weekofyear'] = all_data.DateTime.apply(lambda d: d.strftime('%U')).astype('int')  # [00,53]
all_data['Year'] = all_data.DateTime.apply(lambda d: d.strftime('%y')).astype('int') # two digit year

all_data.Hour.value_counts()
all_data.Ampm.value_counts()
all_data.Dayofweek.value_counts()
all_data.Dayofmonth.value_counts()
all_data.Month.value_counts()
all_data.Weekofyear.value_counts()
all_data.Year.value_counts()


dummies = pd.get_dummies(all_data['Ampm'])
all_data = pd.concat((all_data, dummies), axis=1)

all_data.head()

#---------------------------------------------------------------------------

vec = CountVectorizer(min_df=10)
breeds = vec.fit_transform(all_data['Breed'])
breeds = breeds.toarray()
breeds.shape
#create new features from breed names
for ii in range(breeds.shape[1]):
    colname = 'breed_%d' %ii
    all_data[colname] = pd.Series(breeds[:,ii])

#--------------------------------------------------------------------------------------------------


vec = CountVectorizer(min_df=10)
colors = vec.fit_transform(all_data['Color'])
colors = colors.toarray()
colors.shape
#create new features from breed names
for ii in range(colors.shape[1]):
    colname = 'color_%d' %ii
    all_data[colname] = pd.Series(colors[:,ii])



#--------------------------------------------------------------------------------------------------


[item for item in all_data.columns]

#--------------------------------------------------------------------------------------------------



# Lets break things back up into our test and train data sets.
train_data = all_data.iloc[:train_num]
train_labels = train_data.OutcomeType
test_data = all_data.iloc[train_num:]
test_ids = all_data['ID'][train_num:].values


train_data.shape, test_data.shape

#drop the columns we don't need
cols_recoded = [
 'AgeuponOutcome',
 'AnimalType',
 'Breed',
 'Color',
 'DateTime',
 'Name',
 'SexuponOutcome',
 'Agecategory',
 'Breedtype',
 'Colortype',
 'unknown',
 'Ampm']


train_data = train_data.drop(cols_recoded, axis=1)
test_data = test_data.drop(cols_recoded, axis=1)

train_data.head()

[item for item in train_data.columns]

other_cols = [
  'ID',
 'OutcomeSubtype',
 'OutcomeType']
train_data = train_data.drop(other_cols, axis=1)
test_data = test_data.drop(other_cols, axis=1)


----------------------------------------------------------------------

#visualize the trainining data in 2d
pca = PCA()
pca.fit(train_data)
print "Total Explained variance for the first 50 components:"
res = np.cumsum(pca.explained_variance_ratio_)
print res[:50]

pca2d = PCA(n_components=2)
train_2d = pca2d.fit_transform(train_data)

cm_bright = ListedColormap(['red', 'orange', 'blue', 'white', 'black'])
plt.figure(figsize=(12, 4))
plt.scatter(train_2d[:,0], train_2d[:,1], c=train_labels.cat.codes, cmap=cm_bright)

----------------------------------------------------------------------

train_data['OutcomeType'] = train_labels

#train- test set split
trainset, testset, trainset_labels, testset_labels = \
    cross_validation.train_test_split(train_data, train_labels, test_size=0.2)
    
print trainset.shape
print testset.shape
print trainset_labels.shape
print testset_labels.shape



-----------------------------------------

#group by outcome type

outcome_groups = trainset.groupby('OutcomeType')
for name, group in outcome_groups:
    print name, "\t\t", len(group)

adoption_grp = outcome_groups.get_group('adoption')
died_grp = outcome_groups.get_group('died')
euth_grp = outcome_groups.get_group('euthanasia')
return_grp = outcome_groups.get_group('return_to_owner')
transfer_grp = outcome_groups.get_group('transfer')

#drop the outcometype column again; else the models will choke on string data
trainset = trainset.drop('OutcomeType', axis=1)
testset = testset.drop('OutcomeType', axis=1)

adoption_grp = adoption_grp.drop('OutcomeType', axis=1)
died_grp = died_grp.drop('OutcomeType', axis=1)
euth_grp = euth_grp.drop('OutcomeType', axis=1)
return_grp = return_grp.drop('OutcomeType', axis=1)
transfer_grp = transfer_grp.drop('OutcomeType', axis=1)

pca = PCA(n_components=5)
pca.fit(trainset)

adoption_grp_pca = pca.transform(adoption_grp)
died_grp_pca = pca.transform(died_grp)
euth_grp_pca = pca.transform(euth_grp)
return_grp_pca = pca.transform(return_grp)
transfer_grp_pca = pca.transform(transfer_grp)

testset_pca = pca.transform(testset)


def P5():
    #project train and test data into 2D
   
    #fit a GMM on the positive and negative training data
    gmm_adoption = GMM(n_components=4, covariance_type='full')
    gmm_adoption.fit(adoption_grp_pca)

    gmm_died = GMM(n_components=4, covariance_type='full')
    gmm_died.fit(died_grp_pca)

    gmm_euth = GMM(n_components=4, covariance_type='full')
    gmm_euth.fit(euth_grp_pca)

    gmm_return = GMM(n_components=4, covariance_type='full')
    gmm_return.fit(return_grp_pca)

    gmm_transfer = GMM(n_components=4, covariance_type='full')
    gmm_transfer.fit(transfer_grp_pca)
    
    testset_proba = np.zeros((len(testset_labels),len(set(testset_labels))))    
    
    #calculate the probabily of the predicted label on test data
    testset_proba[:,0] = gmm_adoption.score(testset_pca)
    testset_proba[:,1] = gmm_died.score(testset_pca)
    testset_proba[:,2] = gmm_euth.score(testset_pca)
    testset_proba[:,3] = gmm_return.score(testset_pca)
    testset_proba[:,4] = gmm_transfer.score(testset_pca)

    testset_proba = np.exp(testset_proba)

    print "Log Loss:", metrics.log_loss(testset_labels, testset_proba)  
    testset_pred = [labels[row.argmax()] for row in testset_proba]
    print metrics.classification_report(testset_labels, testset_pred)

