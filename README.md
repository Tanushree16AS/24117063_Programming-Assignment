# For numerical operations
import numpy as np
# For data manipulation
import pandas as pd
# For Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# For statistics
from scipy.stats import zscore
# For Fitting and Transforming data # Majorly used for ML Models
from sklearn.impute import SimpleImputer

# Loading the file
df = pd.read_excel('Career data_PDA_4053.xlsx')

# Column information (non-null count and data type)
print(df.info())
print('='*75)

# Describes the data
print(df.describe())
print('='*75)

# Display few rows
print(df.head())
print('='*75)

# Rows and Columns
print('Shape:',df.shape)
print('='*75)

# Missing Values
print('Missing Values\n')
print(df.isnull().sum())
print('='*75)

# Total Null values in Data Frame
Total_Null_Count = df.isnull().sum().sum()
print('Total Null Count:', Total_Null_Count)
print('='*75)

# Duplicate rows
print('Duplicated Values:',df.duplicated().sum())
print('='*75)

# Create a copy of initial data
df_ori = df.copy()
#print(df_ori.describe())

# Cleaning the data and correcting Numerical columns
df['Salary'] = pd.to_numeric(df['Salary'], errors = 'coerce')
df['Career Change Interest'] = pd.to_numeric(df['Career Change Interest'], errors = 'coerce')
df['Certifications'] = pd.to_numeric(df['Certifications'], errors = 'coerce')
df['Geographic Mobility'] = pd.to_numeric(df['Geographic Mobility'], errors = 'coerce')

# Dropping the rows with all null values
df.dropna(how = 'all', inplace = True)

# Mapping the elements by using .map()function - Used in ML model training
# Gender {Male: 0, Female: 1}
# if df['Gender'].dtype == 'O':
#    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
# Industry growth rate {High: 1, Medium: 0, Low: -1}
# if df['Industry Growth Rate'].dtype == 'O':
#     df['Industry Growth Rate'] = df['Industry Growth Rate'].map({'High': 1, 'Medium': 0, 'Low': -1})
# Education level {High School: 1, Bachelor's: 2, Master's: 3, PhD: 4}
# if df['Education Level'].dtype == 'O':
 # df['Education Level'] = df['Education Level'].map({'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4})

 # Numeric Columns
num_imputer = SimpleImputer(strategy='median')

# Mode Imputer
mode_imputer = SimpleImputer(strategy='most_frequent')

# Categorical Columns
mode_cols = ["Gender","Industry Growth Rate","Education Level","Current Occupation","Field of Study"]

numeric_cols = ["Age","Years of Experience","Salary","Job Opportunities","Skills Gap",'Career Change Interest',
                'Certifications','Geographic Mobility','Job Security','Work-Life Balance','Freelancing Experience',
                'Career Change Events','Professional Networks','Technology Adoption',"Job Satisfaction"]
target = ["Likely to Change Occupation"]

df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
df[mode_cols] = mode_imputer.fit_transform(df[mode_cols])

df[target] = mode_imputer.fit_transform(df[target])

# Resetting the index, as pandas does not update the indexes after dropping rows
df = df.reset_index(drop = True)

# Rows and Columns after cleaning
print('Shape:',df.shape)
print('='*75)

# Missing Values after cleaning in column form
print('Missing Values\n')
print(df.isnull().sum())
print('='*75)

# Total count of all missing values after cleaning
print('Total Missing Values:',df.isnull().sum().sum())
print('='*75)

# Column information (non-null count and data type)
print(df.info())
print('='*75)

# Create a copy after cleaning
df_cleaned = df.copy()
#print(df_cleaned.describe())

#Calculate Z Score to find Outliers
ZScore = pd.DataFrame()
for i in df.columns:
  if df[i].dtype !='O':
    ZScore[i] = zscore(df[i])
print(ZScore)
print('='*75)

outlier_values = {}
for i in ZScore.columns:
  Outliers = ZScore[abs(ZScore[i])>3]
  outlier_values[i] = len(Outliers)
for i in outlier_values.keys():
  print(i,':', outlier_values[i])

  # Correlation of each column with 'Likely to Change Occupation' column
correlation = df.corr(numeric_only = True)
correlation['Likely to Change Occupation'].sort_values()

# Correlation Heatmap
num_cols = df.select_dtypes('number').columns
plt.figure(figsize=(15, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Cross-Tables
pd.crosstab(df['Gender'],df['Likely to Change Occupation'])

pd.crosstab(df['Salary'],df['Likely to Change Occupation'])

pd.crosstab(df['Education Level'],df['Career Change Interest'])

pd.crosstab(df['Current Occupation'],df['Career Change Interest'])

pd.crosstab(df['Gender'],df['Career Change Interest'])

# Target Variable
target = 'Likely to Change Occupation'

num_cols = df.select_dtypes('number').columns
for col in num_cols:
    #sns.boxplot(x=target, y=col, data=df)
    fig = px.box(df, target, col,title = f'{col} vs Likely to Change Occupation' )
    fig.show()

# Plot
plt.figure(figsize=(10,6))
plt.scatter(x=df['Age'], y=df['Likely to Change Occupation'])
plt.xlabel('Age')
plt.ylabel('Likely to Change Occupation')
plt.title('Scatterplot of Age vs. Likelihood to Change Occupation')

# Plot
px.scatter(
    df,
    x='Salary',
    y='Likely to Change Occupation',
    color='Salary',  # This groups and colors the points by the value in the 'Age' column
    title='Scatter Plot Grouped by Raw Age'
)

# Plot
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

print("\n===== Univariate Analysis =====")

## Numeric Variables
for col in num_cols:
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=10, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()

# Plot
df.groupby('Career Change Interest')['Years of Experience'].mean()
plt.scatter(x='Career Change Interest', y='Years of Experience', data=df)
plt.show()

# Plot
print("\n===== Bivariate Analysis =====")
target = 'Likely to Change Occupation'

## Categorical vs Target (Mean target comparison)
for col in cat_cols:
    plt.figure(figsize=(6, 4))
    df.groupby(col)[target].mean().plot(kind='bar', color='lightgreen', edgecolor='black')
    plt.title(f'{col} vs {target}(Mean)')
    plt.ylabel('Mean Target Value')
    plt.grid(alpha=0.3)
    plt.show()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Education Level', y='Salary', data=df, estimator=np.mean)
plt.title('Salary Distribution by Education Level')
plt.show()

# Plot
plt.figure()
plt.hist(df['Years of Experience'], bins=15, edgecolor = 'black')
plt.title("Histogram")
plt.xlabel('Age')
plt.ylabel("Likely to Change Occupation")
plt.show()

# Plot
plt.figure(figsize =(7,7))
plt.pie(df['Likely to Change Occupation'].value_counts(), labels=['No', 'Yes'], autopct='%1.1f%%', startangle=90)
plt.title("Overall% of Likely to Channge Occupation ")
plt.show()
