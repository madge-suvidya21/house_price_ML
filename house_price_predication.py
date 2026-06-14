#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Load Dataset
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.shape)
train.head()


# In[2]:


#Check Missing Values
missing = train.isnull().sum()
missing = missing[missing > 0]
print(missing.sort_values(ascending=False))


# In[3]:


# Data Preprocessing
# Separate Features and Target

X = train.drop("SalePrice", axis=1)
y = train["SalePrice"]


# In[4]:


#Handle Missing Values

X = X.fillna(X.median(numeric_only=True))


# In[5]:


#Encode Categorical Variables

X = pd.get_dummies(X)


# In[6]:


# Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[7]:


# Train Multiple Models

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[8]:


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}


# In[9]:


#Compare Models

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import Lasso


results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append([name, r2, mae, rmse])

results_df = pd.DataFrame(
    results,
    columns=["Model", "R2 Score", "MAE", "RMSE"]
)

print(results_df.sort_values(by="R2 Score", ascending=False))


# In[10]:


print(results_df.sort_values(by='R2 Score', ascending=False))


# In[11]:


# Results in Visulization Format
import matplotlib.pyplot as plt

results_df = results_df.sort_values(by='R2 Score', ascending=False)

plt.figure(figsize=(10,5))
plt.bar(results_df['Model'], results_df['R2 Score'])
plt.title('Model Comparison - R² Score')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt

results_df = results_df.sort_values(by='R2 Score', ascending=False)

plt.figure(figsize=(10,5))
plt.bar(results_df['Model'], results_df['R2 Score'])
plt.title('Model Comparison - R² Score')
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[13]:


plt.figure(figsize=(10,5))
plt.bar(results_df['Model'], results_df['MAE'])
plt.title('Model Comparison - MAE')
plt.xlabel('Models')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[14]:


best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# In[15]:


best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, preds)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()


# In[16]:


import pandas as pd

best_model = RandomForestRegressor(random_state=42)
best_model.fit(X_train, y_train)

importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': best_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
).head(10)

plt.figure(figsize=(10,6))
plt.barh(
    importance['Feature'],
    importance['Importance']
)
plt.title('Top 10 Important Features')
plt.tight_layout()
plt.show()


# In[ ]:




