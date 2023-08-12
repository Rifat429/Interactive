#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd


#import libraries
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import re
from sklearn import metrics
import seaborn as sn
import lightgbm as lgb


# In[2]:


df=pd.read_csv('C:/Users/User/OneDrive/Desktop/Cakri/archive/Resume/Resume.csv')
df


# In[3]:


df=df.drop(['ID', 'Resume_html'],axis=1)


# In[4]:


df


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


# In[6]:


tfidf = TfidfVectorizer(analyzer = 'word', max_features = 150)
X = tfidf.fit_transform(df['Resume_str'])
X


# In[7]:


y = df['Category']
X = X.toarray()
X


# In[8]:


X = pd.DataFrame(X)
X


# In[9]:


feature = tfidf.vocabulary_
col_names = []


# In[10]:


for key, value in feature.items():
    print(key, ' : ', value)
    col_names.append(key)


# In[11]:


pot_lbl = df["Category"].value_counts()

# Barplot
plt.figure(figsize=(8,5))
sns.barplot(x=pot_lbl.index, y=pot_lbl)
plt.xlabel('result', fontsize=15)
plt.ylabel('count', fontsize=15)


# In[12]:


X.columns = col_names
col_names
X


# In[13]:


#correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(X.corr(),annot=True)
plt.title('Correlation Matrix',fontsize=20)


# In[22]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print("Model:", model.__class__.__name__)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate and plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# In[ ]:


# Define a list of models to evaluate
models = [LogisticRegression(),SVC(),GaussianNB(),KNeighborsClassifier(),RandomForestClassifier(), DecisionTreeClassifier(),HistGradientBoostingClassifier(),ExtraTreesClassifier(),MLPClassifier()]

for model in models:
    train_and_evaluate_model(model,  X_train, y_train, X_test, y_test)

    


# In[17]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import log_loss, cohen_kappa_score, jaccard_score
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix, classification_report
fold=10
# i=0

# 'Network attacks'
# 'Malware attacks'
# 'Exploit attacks'
# 'Reconnaissanc attacks'
# 'Testing Evaluation attacks'
# 'Normal'

model=RandomForestClassifier()


# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


# Create an empty list to store the accuracy scores for each fold
accuracy_scores = []

# Create empty arrays to store the confusion matrices and classification reports for each fold
confusion_matrices = []



# Initialize an empty dictionary to store classification reports
reports = {}
# Initialize an empty DataFrame to store the mean classification report
mean_report = pd.DataFrame()



for train_index, test_index in skf.split(X, y):
    
    # Split the data into training and testing sets
    X_train, X_test =  X.iloc[train_index], X.iloc[test_index]
    y_train, y_test =   y.iloc[train_index],  y.iloc[test_index]
    
    model.fit(X_train,y_train)
    
    y_pred=model.predict(X_test)
    
    acc = accuracy_score(y_test,y_pred)
    
    
    
    report= classification_report(y_test,y_pred,output_dict=True)
    
  
    
    # Calculate the confusion matrix and classification report for the current fold
    cm = confusion_matrix(y_test, y_pred)
#     print(cm)
    
    accuracy_scores.append(acc)
    # Append the classification report to the dictionary
    reports[fold] = report
    # Append the confusion matrix and classification report to their respective arrays
    confusion_matrices.append(cm)
    
#     i=i+1


# Concatenate the classification reports into a single DataFrame
mean_report = pd.concat([pd.DataFrame.from_dict(reports[i]) for i in reports], axis=1)

# Compute the mean of the classification report DataFrame
mean_report = mean_report.mean(axis=1, level=0)

# Display the mean classification report
print(mean_report)

# Calculate the average confusion matrix and classification report across all folds
avg_cm = np.mean(confusion_matrices, axis=0)

# Calculate the average accuracy score across all folds
avg_accuracy = np.mean(accuracy_scores)
print("Average Accuracy Score:", avg_accuracy)

import seaborn as sns
import matplotlib.pyplot as plt
#np.set_printoptions(suppress=True)
avg_cm_ceil = np.ceil(avg_cm)
#sns.set(font_scale=1.4)

avg_cm_ceil = avg_cm_ceil.astype(int)

class_labels=['HR', 'DESIGNER', 'INFORMATION-TECHNOLOGY', 'TEACHER', 'ADVOCATE',
       'BUSINESS-DEVELOPMENT', 'HEALTHCARE', 'FITNESS', 'AGRICULTURE',
       'BPO', 'SALES', 'CONSULTANT', 'DIGITAL-MEDIA', 'AUTOMOBILE',
       'CHEF', 'FINANCE', 'APPAREL', 'ENGINEERING', 'ACCOUNTANT',
       'CONSTRUCTION', 'PUBLIC-RELATIONS', 'BANKING', 'ARTS', 'AVIATION']
# Generate heatmap of average confusion matrix
sns.heatmap(avg_cm_ceil, annot=True, cmap='Blues',fmt='d',xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


rf_df = pd.DataFrame(mean_report).transpose()
rf_plot=sns.heatmap(avg_cm_ceil, annot=True, cmap='Blues',fmt='d',xticklabels=class_labels, yticklabels=class_labels)
   


# In[21]:


# Concatenate the dataframes into a single dataframe
reports_df = pd.concat([rf_df])
# Save the plots to a file
rf_plot.figure.savefig('rf_cm.png')




# Save the dataframes to an Excel file
writer = pd.ExcelWriter('categorized_resumes.xlsx', engine='xlsxwriter')
reports_df.to_excel(writer, sheet_name='Reports')
workbook = writer.book

# Add the confusion matrix plots to the Excel file
worksheet = writer.sheets['Reports']
worksheet.insert_image('I2', 'rf_cm.png')

# Close the writer
writer.save()


# In[20]:


# get_ipython().system('pip install xlsxwriter')


# In[ ]:




