# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





#Code starts here
data = pd.read_csv(path)
data['Rating'].hist()
plt.show()

data = data[data['Rating']<=5]
data['Rating'].hist()
plt.show()
#Code ends here


# --------------
# code starts here
# Finding null values
total_null = data.isnull().sum()
percent_null = (total_null/data.isnull().count())*100
missing_data = pd.concat([total_null,percent_null],axis=1,keys=['Total','Perceent'])
print(missing_data)

# Removing null values
data.dropna(axis = 0 , inplace = True)
total_null_1 = data.isnull().sum()
percent_null_1 = (total_null_1/data.isnull().count())*100
missing_data_1 = pd.concat([total_null_1, percent_null_1], axis = 1, keys = ['Total','Percent'])
print(missing_data_1)
# code ends here


# --------------

#Code starts here

sns.catplot(x="Category",y="Rating",data=data, kind="box", height = 10)
plt.xticks(rotation=90)
plt.title('Rating vs Category [BoxPlot]')
#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data['Installs'])
data['Installs'] = data['Installs'].str.replace(',','')
data['Installs'] = data['Installs'].str.replace('+','')
data['Installs'] = data['Installs'].astype(int)

le = LabelEncoder()
le.fit(data['Installs'])
data['Installs'] = le.transform(data['Installs'])
sns.regplot(data=data, x='Installs', y='Rating', )
plt.title('Rating vs Installs [RegPlot]')
#Code ends here


# --------------
#Code starts here
print(data['Price'].value_counts())
data['Price'] = data['Price'].str.replace("$","")
data['Price'] = data['Price'].astype(float)

#Checking Correlation
plt.figure(figsize = (14,10))
a = sns.regplot(x = "Price", y = 'Rating', data = data)
a.axes.set_title('Rating vs Price [RegPlot]', fontsize = 20)
a.set_xlabel("Price", fontsize = 18) 
a.set_ylabel("Rating", fontsize = 18)
a.tick_params(labelsize = 8)

#Code ends here


# --------------

#Code starts here
data['Genres'].unique()
a = []
for i in data['Genres']:
    a.append(i.split(';')[0])
data['Genres'] = a

gr_mean = data.groupby('Genres', as_index=False)['Rating'].mean()
gr_mean.describe()
gr_mean = gr_mean.sort_values(by = 'Rating')
print('first vale of gr_mean:',gr_mean.iloc[0])
print('-'*10)
print('last value of gr_mean:', gr_mean.iloc[-1])
#Code ends here


# --------------

#Code starts here
data['Last Updated']
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).dt.days
sns.regplot(data=data, x='Last Updated Days', y='Rating')
plt.title('Rating vs Last Updated [RegPlot]')
#Code ends here


