#!/usr/bin/env python
# coding: utf-8

# In[9]:


#Import necessary libraries

# Data Handling and Visualization tools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Filter future Warnings
import warnings
warnings.filterwarnings("ignore")

# Data Preprocessing tools
from sklearn.preprocessing import MinMaxScaler

# Model Training and Evaluation
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score, f1_score, precision_score


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from sklearn.pipeline import Pipeline

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score


# In[10]:


import pandas as pd

def load_data(data):
    df = pd.read_csv(data)
    
    return df


# In[11]:


import pandas as pd

def check_data(df):
    """
    Creates a DataFrame that checks for shape, size, info, and describe, and displays the output containing the information.
    
    Parameters:
    df (pd.DataFrame): The dataframe being described.
    
    Returns:
    None: Prints the description.
    """
    
    # Print shape of the DataFrame
    print("Shape of DataFrame:")
    print(df.shape)

    # Print size of the DataFrame
    print("Size of DataFrame:")
    print(df.size)

    # Print the info of the DataFrame
    print("Info of DataFrame:")
    print(df.info())

    # Print the descriptive statistics of the DataFrame
    print("Descriptive Statistics:")
    print(df.describe().T)


# In[12]:


import pandas as pd
def analyze_data(df):
    

    print("Missing Values:\n", df.isnull().sum())

    print("Duplicate Rows:\n", df.duplicated().sum())


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def outlierzz_data(df):
    """
    Plots box plots for each numerical column in the DataFrame to check for outliers.

    Parameters:
    df (pd.DataFrame): The dataframe being analyzed.
    
    Returns:
    None: Displays box plots for each numerical column.
    """
    # Check for outliers using boxplots
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    if len(numerical_cols) > 0:
        plt.figure(figsize=(12, len(numerical_cols) * 4))
        for i, col in enumerate(numerical_cols):
            plt.subplot(len(numerical_cols), 1, i + 1)
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
        
        plt.tight_layout()
        plt.savefig('Outliers')
        plt.show()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

def univariate_analysis(df):
    '''Perform univariate analysis on all columns and produce a single output with all plots'''
    
    num_columns = df.shape[1]
    num_rows = num_columns // 4 + (num_columns % 4 > 0)
    
    fig, axes = plt.subplots(num_rows, 4, figsize=(20, 5 * num_rows))
    axes = axes.flatten()
    
    for i, column in enumerate(df.columns):
        if df[column].dtype == 'object':
            sns.countplot(x=df[column], ax=axes[i])
            axes[i].set_ylabel('Count')
        else:
            sns.histplot(df[column], kde=True, ax=axes[i])
            axes[i].set_xlabel(column)
        
        axes[i].set_title(column)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('Univariate Analysis')
    plt.show()


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

def bivariate_analysis(df, target="churn"):
    '''Perform bivariate analysis for all columns with respect to the target variable'''
    
    # Convert boolean or categorical target column to string for Seaborn compatibility
    if df[target].dtype == 'bool':
        df[target] = df[target].astype(str)
    elif df[target].dtype == 'category':
        df[target] = df[target].astype(str)

    num_cols = len(df.columns) - 1  # Exclude the target column
    num_rows = (num_cols // 3) + (num_cols % 3 > 0)  # Determine the number of rows needed

    plt.figure(figsize=(20, 4 * num_rows))
    
    for i, column in enumerate(df.columns):
        if column == target:
            continue
        plt.subplot(num_rows, 3, i + 1)  # Arrange plots in a grid of 3 columns
        
        if df[column].dtype == 'object':
            sns.countplot(x=column, hue=target, data=df)
        else:
            sns.histplot(df, x=column, hue=target, kde=True, element='step')
        
        plt.title(f'{column} vs {target}')
    
    plt.tight_layout()
    plt.savefig('Bivariate Analysis')
    plt.show()


# In[16]:


def multivariate_analysis(df):
    '''Calculate and plot the correlation matrix for the given dataset.'''


    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))

    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

    # Set the title
    plt.title('Correlation Matrix')

    # Show the plot
    plt.savefig('Multivariate Analysis')
    plt.show()


# In[ ]:




