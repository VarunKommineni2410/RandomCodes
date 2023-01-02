#!/usr/bin/env python
# coding: utf-8

# <b><h2> CROP YIELD PREDICTION IN INDIA </h2></b>
# 
# Predicting yield helps the state to get an estimate of the crop in a
# certain year to control the price rates.This model focuses on predicting the crop yield in advance by analyzing
# factors like location, season, and crop type  through machine learning techniques on
# previously collected datasets.

# In[1]:


# importing necessary libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# loading the dataset

crop_data=pd.read_csv("crop_production.csv")
crop_data


# In[3]:


crop_data.shape

#rows X columns


# In[4]:


# dataset columns
crop_data.columns


# In[5]:


# statistical inference of the dataset

crop_data.describe()


# In[6]:


# Checking missing values of the dataset in each column
crop_data.isnull().sum()


# In[7]:


# Dropping missing values 
crop_data = crop_data.dropna()
crop_data


# In[8]:


#checking
crop_data.isnull().values.any()


# In[9]:


# Displaying State Names present in the dataset
crop_data.State_Name.unique()


# In[10]:


# Adding a new column Yield which indicates Production per unit Area. 

crop_data['Yield'] = (crop_data['Production'] / crop_data['Area'])
crop_data.head(10) 


# In[11]:


# Visualizing the features

ax = sns.pairplot(crop_data)
ax


# In[12]:


# Dropping unnecessary columns

data = crop_data.drop(['State_Name'], axis = 1)


# In[13]:


data.corr()


# In[14]:


sns.heatmap(data.corr(), annot =True)
plt.title('Correlation Matrix')


# In[15]:


dummy = pd.get_dummies(data)
dummy


# <b><i> Splitting dataset into train and test dataset </i></b>

# In[16]:



from sklearn.model_selection import train_test_split

x = dummy.drop(["Production","Yield"], axis=1)
y = dummy["Production"]

# Splitting data set - 25% test dataset and 75% 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25, random_state=5)

print("x_train :",x_train.shape)
print("x_test :",x_test.shape)
print("y_train :",y_train.shape)
print("y_test :",y_test.shape)


# In[17]:


print(x_train)
print(y_train)


# <b><h3> Linear Regression </b></h3>

# In[18]:


# Training the Simple Linear Regression model .

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[19]:


# Predicting the test Results 

lr_predict = model.predict(x_test)
lr_predict


# In[20]:


model.score(x_test,y_test)


# In[21]:


from sklearn.metrics import r2_score
r = r2_score(y_test,lr_predict)
print("R2 score : ",r)


# In[22]:


plt.scatter(y_test,lr_predict)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression')


# Clearly, the dataset is not good for linear regression.
# 
# <b> Assumptions of Linear Regression </b>
# <ol>
#     <li> Linearity.</li>
#     <li> Homoscedasticity </li>
#     <li> Multivariate normality </li>
#     <li> Lack of multicollinearity </li>
#     
# 

# # R2 score: This is pronounced as R-squared, and this score refers to the coefficient of determination. 
# # This tells us how well the unknown samples will be predicted by our model.

# <b><h3> Random Forest Algorithm </h3></b>

# In[32]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 11)
model.fit(x_train,y_train)
rf_predict = model.predict(x_test)
rf_predict


# In[29]:


model.score(x_test,y_test)


# In[57]:


# Calculating R2 score

from sklearn.metrics import r2_score
r1 = r2_score(y_test,rf_predict)
print("R2 score : ",r1)


# In[58]:


# Calculating Adj. R2 score: 

Adjr2_1 = 1 - (1-r)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_1))


# In[31]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(rf_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Random Forest Regression')


# <b> Comparison between Linear Regression Algorithm and Random Forest Algorithm </b> 

# 
# 
# 1. Linear regression algorithm is not at all accurate for this kind of prediction.
# 2. Random Forest Algorithm has higher accuracy ( between 85 % to 90% ), but it is slow.

# <b> Support Vector Regression </b> 

# In[ ]:


# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# In[ ]:


print(x_train)
print(x_test)


# In[ ]:


# Training the SVR model 

from sklearn.svm import SVR 
regressor = SVR(kernel = 'rbf')
regressor.fit(x_train,y_train)


# In[ ]:


# Predicting Result

svr_predict = regressor.predict(x_test)
svr_predict


# In[ ]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(svr_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Support Vector Regression')


# <b> Decision Tree </b>

# In[33]:


# Training model 
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 5)
regressor.fit(x_train,y_train)

# Predicting results
decisiontree_predict = regressor.predict(x_test)
decisiontree_predict


# In[34]:


regressor.score(x_test,y_test)


# In[60]:


# Calculating R2 score :

from sklearn.metrics import r2_score
r2 = r2_score(y_test,decisiontree_predict)
print("R2 score : ",r2)


# In[56]:


# Calculating Adj. R2 score: 

Adjr2_2 = 1 - (1-r)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
print("Adj. R-Squared : {}".format(Adjr2_2))


# In[37]:


ax = sns.distplot(y_test, hist = False, color = "r", label = "Actual value ")
sns.distplot(decisiontree_predict, hist = False, color = "b", label = "Predicted Values", ax = ax)
plt.title('Decision Tree Regression')


# <b> Cross-validation </b> 

# In[38]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = x_train, y=y_train, cv = 10)


# In[39]:


a1 = (accuracies.mean()*100)
b1 = (accuracies.std()*100)


# In[40]:



# Mean Accuracy and SD of 10 fold results

print("Accuracy : {:.2f}%".format (accuracies.mean()*100))
print("Standard Deviation : {:.2f}%".format(accuracies.std()*100))


# <b> Cross-validation </b>

# In[36]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y=y_train)


# In[42]:


a2 = (accuracies.mean()*100)
b2 = (accuracies.std()*100)


# In[43]:


print("Accuracy : {:.2f}%".format (accuracies.mean()*100))
print("Standard Deviation : {:.2f}%".format(accuracies.std()*100))


# In[46]:


import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [a1, a2]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Accuracy(in %)')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()


# In[52]:


import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [b1, b2]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Standard Deviation(in %)')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[62]:


import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [Adjr2_1, Adjr2_2]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Standard Deviation(in %)')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[65]:


import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [r1, r2]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('R-Squared Score')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[64]:


import numpy as np
import matplotlib.pyplot as plt
 
# create a dataset
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [Adjr2_1, Adjr2_2]

x_pos = np.arange(len(Accuracy))

# Create bars with different colors
plt.bar(x_pos, Accuracy, color=['#488AC7','#ff8c00'])

# Create names on the x-axis
plt.xticks(x_pos, Algorithms)
plt.ylabel('Adjusted R-Squared Score')
plt.xlabel('Machine Learning Regression Techniques')

# Show graph
plt.show()
plt.savefig('SD.png')


# In[ ]:


mae = metrics.mean_absolute_error(y_test, y_pred)


# In[49]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(rf_predict,y_test))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, rf_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, rf_predict)))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(8, 5))
 
# set height of bar
Algorithms = ['Random Forest', 'Decision-tree']
Accuracy = [a1, a2]
Standard_Deviation = [b1,b2]
 
# Set position of bar on X axis
br1 = np.arange(len(Accuracy))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Accuracy, color ='blue', width = barWidth,
        edgecolor ='grey', label ='Accuracy')
plt.bar(br2, Standard_Deviation, color ='maroon', width = barWidth,
        edgecolor ='grey', label ='Standard Devation')
 
# Adding Xticks
plt.xlabel('Algorithms', fontweight ='bold', fontsize = 10)
plt.ylabel('Accuracy (in %)', fontweight ='bold', fontsize = 10)
plt.xticks([r + barWidth for r in range(len(Accuracy))],
        Algorithms)
 
plt.legend()
plt.show()


# <b> Hyperparameter Tuning using GridSearchCV </b>

# Random Forest Regression 

# In[37]:


from sklearn.model_selection import GridSearchCV

# defining parameter range 
param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'gamma':['auto'],
              'kernel': ['rbf','linear']}  
   
reg = GridSearchCV(DecisionTreeRegressor(), param_grid, refit = True, verbose = 3,n_jobs=-1) 
   
reg.fit(x_train,y_train)
reg.grid_scores_


# In[ ]:


# CV results are not easy to use, 
# sklearn provides a way to download these results into a dataframe 
df = pd.DataFrame(reg.cv_results_)
df


# In[ ]:


df[['param_C','param_kernel','mean_test_score']]


# In[ ]:


reg.best_score_


# In[ ]:


reg.best_params_


# In[ ]:


# To tackle the computation problem in gridsearch , 
# randomizedsearchcv comes in. Randomly tries value.


# In[39]:


from sklearn.model_selection import RandomizedSearchCV
rs = RandomizedSearchCV(regressor(gamma='auto'),{
    'c': [1,10,20],
    'kernel' : ['rbf','linear']
},
 cv = 5,
 return_train_score=False,
 n_iter=2
)
rs.fit(x_train,y_train)
pd.DataFrame(rs.cv_results_)[['param_C','param_kernel','mean_test_score']]


# Decision Tree
# 

# In[ ]:





# In[ ]:




