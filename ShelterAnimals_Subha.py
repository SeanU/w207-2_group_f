
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np

from datetime import date, datetime


# In[2]:

train = pd.read_csv('C:\\Subha\\WS207-ML\\Kaggle\\git\\train.csv', parse_dates=['DateTime'])
test = pd.read_csv('C:\\Subha\\WS207-ML\\Kaggle\\git\\test.csv', parse_dates=['DateTime'])


# In[3]:

train.describe()
train.dtypes
train.columns
train.index

train.head()

sum(train['Breed'] == None)
sum((train['Color'] == "") | (train['Color'] == None) )

#almost uniform distribution
train['DateTime'].hist()
# In[4]:

test.head(2)


# ### Some utility functions for assisting in feature creation.
# #### For the following unknowns are gettting recoded to -999

# In[5]:

# From: http://stackoverflow.com/a/28688724
Y = 2000 # dummy leap year to allow input X-02-29 (leap day)
seasons = [('winter', (date(Y,  1,  1),  date(Y,  3, 20))),
           ('spring', (date(Y,  3, 21),  date(Y,  6, 20))),
           ('summer', (date(Y,  6, 21),  date(Y,  9, 22))),
           ('autumn', (date(Y,  9, 23),  date(Y, 12, 20))),
           ('winter', (date(Y, 12, 21),  date(Y, 12, 31)))]

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()
    now = now.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= now <= end)


#exploring breed categories
train[train['Breed'] == 'Black/Tan Hound Mix']
#Black/Tan Hound Mix - about 12
[s for s in train['Breed'] if ('/' in s)  & ('mix' in s.lower())]
#Jack Russell Terrier/Unknown - 1
[s for s in train['Breed'] if ('/' in s)  & ('known' in s.lower())]
#unknown mix
[s for s in train['Breed'] if ('/' not in s)  & ('known' in s.lower())]
#<breed> mix
[s for s in train['Breed'] if ('/' not in s)  & ('mix' in s.lower())]


def get_breed(line):
    #special case this miscoded value : should be Black and Tan hound Mix (Black and Tan hound is a breed)
    if line == 'Black/Tan Hound Mix':
        return 'purebreed_mix'
        
    line = line.lower()
    
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


pd.isnull(train['Color']).any()
[s for s in train['Color'] if ('/' in s)  & ('mix' in s.lower())]
#Jack Russell Terrier/Unknown - 1
[s for s in train['Color'] if ('/' in s)  & ('known' in s.lower())]
#unknown mix
[s for s in train['Breed'] if ('/' not in s)  & ('known' in s.lower())]
#<breed> mix
[s for s in train['Breed'] if ('/' not in s)  & ('mix' in s.lower())]


def get_color(x):
    if pd.isnull(x):
        return x
        
    colors = x.strip().split('/')
    if len(colors) == 1:
        return "single_color"
    elif len(colors) == 2:
        return "double_color"
    else:
        return "lots_color" 
        


def age_years(x):
    if(pd.isnull(x)):
        return x
    
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




def gender(string):
    try:
        if 'Female' in string:
            return 1

        if 'Male' in string:
            return 0
        
        return -999

    except:
        return -999
    
def fixed(string):
    try:
        if 'Spayed' in string or 'Neutered' in string:
            return 1

        if 'Unknown' in string:
            return -999

        return 0
    except:
        return -999


# ### We are going to modify all the data test+train together to generate our new features then break them back up for simplicity sake.

# In[6]:

# get number of training elements
train_num = train.shape[0]

#df.rename(columns={'$a': 'a', '$b': 'b'}, inplace=True)
train.rename(columns={'AnimalID': 'ID'}, inplace=True)

all_data = pd.concat((train, test), axis=0, ignore_index=True)

# create a new data frame to store our new features.
new_data = pd.DataFrame()

#We need an index to work with
new_data['ID'] = all_data['ID']

#does it have a name?
new_data['has_name'] = all_data['Name'].map(lambda x: 1 if not pd.isnull(x) and (x).strip() else 0)
new_data['has_name'].value_counts()

# Add the easy stuff to our new dataframe
# is it a cat?
pd.isnull(all_data['AnimalType']).any()
all_data['AnimalType'].str.contains('known').any()
new_data['is_cat'] = all_data['AnimalType'].map(lambda x: 1 if 'Cat' in x else 0) 
new_data['is_cat'].value_counts()

#breed type
new_data['breed_type'] = all_data['Breed'].map(get_breed).astype('category')
new_data['breed_type'].value_counts()

new_data['day_of_week'] = all_data['DateTime'].map(lambda x: x.strftime("%A")).astype('category')
new_data['day_of_week'].value_counts()

new_data['color_type'] = all_data['Color'].map(get_color).astype('category')
new_data['color_type'].value_counts()

#TODO: add in month, season, time_of_day




# color stuff
#new_data = new_data.join(all_data['Color'].apply(
#        lambda x: pd.Series({'color_1':x.split('/')[0], 'color_2':x.split('/')[1]}
#                            if len(x.split('/')) == 2 else {'color_1':x, 'color_2':'NaN'})))
#
## lets convert the date into seasons
#new_data = new_data.join(
#    pd.get_dummies(
#        all_data['DateTime'].map(lambda x: get_season(datetime.strptime(x, '%Y-%m-%d %H:%M:%S')))
#    ))





# In[8]:
pd.isnull(all_data['AgeuponOutcome']).any()
all_data[pd.isnull(all_data['AgeuponOutcome'])]['AgeuponOutcome']
all_data['AgeuponOutcome'].str.contains('known').any()
new_data['age_years'] = all_data['AgeuponOutcome'].map(lambda x: age_years(x))
new_data['age_years'].value_counts()

#puppy - under 1; adolescent - under 2; adult - under 7; senior- 7+
#source - http://www.akc.org/learn/family-dog/how-to-calculate-dog-years-to-human-years/
age_ranges2=[0.0, 0.99, 1.99, 6.99, 29.99]
age_labels=['puppy', 'adolescent', 'adult', 'senior']
new_data['age_type'] = pd.cut(new_data['age_years'], age_ranges2, labels=age_labels).astype('category')
new_data['age_type'].value_counts()



# In[ ]:
pd.isnull(all_data['SexuponOutcome']).any()
# what is the gender, I am coding all unknowns as -999
new_data['is_female'] = all_data['SexuponOutcome'].map(gender)
new_data['is_female'].value_counts()
pd.isnull(new_data['is_female']).any()
# In[ ]:

# are they fixed
new_data['is_fixed'] = all_data['SexuponOutcome'].map(fixed)
new_data['is_fixed'].value_counts()
pd.isnull(new_data['is_fixed']).any()


# ## We are going to recode breed and color so we have a sparse array.
# ### I am also recoding age, but it's probably not the best approach because it is using the quartiles of the two data sets.

# In[ ]:




# In[ ]:


#new_data = new_data.join(
#    pd.get_dummies(
#        new_data['breed_1'], prefix='breed')
#    )

#new_data = new_data.join(
#    pd.get_dummies(
#        new_data['color_1'], prefix='color')
#    )

# This is probably an inappropriate way to dummy code the ages 
# but just to get something together for testing
#new_data = new_data.join(
#    pd.get_dummies(
#        pd.qcut(new_data['age_years'], 4, labels=["age_1","age_2","age_3", "age_4"]))
#    )


# In[ ]:

new_data.head()

new_data = new_data.join(pd.get_dummies(new_data['breed_type'], prefix='breed'))
new_data = new_data.join(pd.get_dummies(new_data['day_of_week'], prefix='day'))
new_data = new_data.join(pd.get_dummies(new_data['color_type'], prefix='color'))
new_data = new_data.join(pd.get_dummies(new_data['age_type'], prefix='age'))


new_data.head()
new_data.dtypes
# In[ ]:

 #We want to drop the original non-binary columns now. 
cols_to_drop = ['breed_type',
                'day_of_week',
                'color_type',
                'age_type',
                'age_years',
                'ID']
new_data = new_data.drop(cols_to_drop, axis=1)

new_data.dtypes

# ### Now we have a binary feature matrix.

# In[ ]:

#new_data.head()


# In[ ]:

# Lets break things back up into our test and train data sets.
X_train_all = new_data.iloc[:train_num]
y_train_all = all_data['OutcomeType'][:train_num]

X_test_all = new_data.iloc[train_num:]
ids_test = all_data['ID'][train_num:].values


# In[ ]:

X_train_all.shape, y_train_all.shape, X_test_all.shape, ids_test.shape


# ### Just to get things going...

# In[ ]:

from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


# In[ ]:

X_train, X_test, y_train, y_test = train_test_split(
    X_train_all, y_train_all, test_size=0.30, random_state=23)


# In[ ]:

lr = LogisticRegression(random_state=23, max_iter=100)


# In[ ]:
#0.6366
y_pred = lr.fit(X_train, y_train).predict(X_test)


# In[ ]:

print np.mean(y_pred == y_test.values)



from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 

clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)
#0.6302
print np.mean(y_pred == y_test.values)

dt = DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=0)
dt.fit(X_train, y_train)

#0.6296
print 'Accuracy (a decision tree):', dt.score(X_test, y_test)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

#0.6312
print 'Accuracy (a random forest):', rfc.score(X_test, y_test)


abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, learning_rate=0.1)
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
print np.mean(y_pred == y_test.values)
#0.6418
#print 'Accuracy (adaboost with decision trees):', abc.score(X_test, y_test)

# ### Not particularly good, but whatever. To create a submission file I think we need the 

# In[ ]:

# Not particularly good, but whatever. To create a submission file I think we need the 
# robability for each class. First we retrain on entire dataset then classify the test data.
y_pred_sub = abc.fit(X_train_all, y_train_all).predict(X_test_all)

rows = {"Adoption":[1,0,0,0,0] , "Died":[0,1,0,0,0], "Euthanasia":[0,0,1,0,0], "Return_to_owner":[0,0,0,1,0], "Transfer":[0,0,0,0,1] }

# ### Prepare the submission file...

# In[ ]:

# Prepare the submission file
sub = pd.DataFrame([rows[y] for y in y_pred_sub], columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
sub.insert(0, 'ID', ids_test.astype(int))


# In[ ]:

sub.head()


# In[ ]:

sub.to_csv("submission2.csv", index=False)


# In[ ]:



# In[ ]:
