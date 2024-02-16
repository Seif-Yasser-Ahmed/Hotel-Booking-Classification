"""Seif-Yasser-Task2.ipynb

##Task1:EDA
"""
from sklearn.preprocessing import OneHotEncoder
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

data = pd.read_csv("first inten project.csv")

data = data.rename(columns={'number of adults': 'Number_Of_Adults',
                            'number of children': 'Number_Of_Children', 'number of weekend nights': 'Number_Of_Weekend_Nights',
                            'number of week nights': 'Number_Of_Week_Nights', 'type of meal': 'Type_Of_Meal', 'car parking space': 'Car_Parking_Space',
                            'room type': 'Room_Type', 'market segment type': 'Market_Segment_Type', 'average price ': 'Average_Price',
                            'special requests': 'Special_Requests', 'booking status': 'Booking_Status'})

data.drop(['Booking_ID'], axis=1, inplace=True)

# data = data.loc[~data.duplicated()]
data = data.reset_index(drop=True)

for i, value in enumerate(data['Number_Of_Week_Nights']):
    if value > 7:
        data.drop(labels=i, inplace=True)
# data = data.loc[~data.duplicated()]
data = data.reset_index(drop=True)
for i, value in enumerate(data['Number_Of_Weekend_Nights']):
    if value > 3:
        data.drop(labels=i, inplace=True)

data["Booking_Status"] = data["Booking_Status"].replace("Not_Canceled", 0)
data["Booking_Status"] = data["Booking_Status"].replace("Canceled", 1)

label_encoder = LabelEncoder()
data['Type_Of_Meal'] = label_encoder.fit_transform(data['Type_Of_Meal'])
data['Room_Type'] = label_encoder.fit_transform(data['Room_Type'])

for i, value in enumerate(data['Number_Of_Adults']):
    if value == 0:
        data['Number_Of_Adults'][i] = np.nan

data.isnull().sum()

# mean_adults=data.mean().iloc[0]
# data=data.fillna(mean_adults)

"""##Some edits on Task1
 *based on recommendation of the report*

 tried to do k-best feature selection but gave the same accuracy

###Splitting date of reservation feature
"""

# To ease the deployment
ohe = OneHotEncoder()
data = pd.get_dummies(data, prefix=['Market_Type'], columns=[
                      'Market_Segment_Type'])

data['date of reservation'] = data['date of reservation'].loc[data['date of reservation'] != '2018-2-29']
data['Date'] = pd.to_datetime(data['date of reservation'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day

data = data.drop(['Date', 'date of reservation'], axis=1)

"""###Outliers for lead time
*Also edited the Average Price outlier handling method*
"""

upp = data['lead time'].mean()+3*data['lead time'].std()
low = data['lead time'].mean()-3*data['lead time'].std()
new_data = data.loc[((data['lead time'] < upp) & (data['lead time'] > low))]
data = new_data
upp = data['Average_Price'].mean()+3*data['Average_Price'].std()
low = data['Average_Price'].mean()-3*data['Average_Price'].std()
new_data = data.loc[((data['Average_Price'] < upp) &
                     (data['Average_Price'] > low))]
data = new_data

for x, value in enumerate(data['lead time']):
    if value < 0:
        data['lead time'][x] = np.nan

data = data.drop('P-not-C', axis=1)
data = data.drop('P-C', axis=1)
data = data.drop('repeated', axis=1)

data.duplicated().sum()

# data = data.loc[~data.duplicated()]

data = data.fillna(data.mean())

data = data.reset_index(drop=True)

data = pd.DataFrame(data)
X = data.drop('Booking_Status', axis=1)
y = data['Booking_Status']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.17, random_state=7)

"""###Standarization
*tried MinMax but gave worse results*
"""

s = StandardScaler()
X_train = s.fit_transform(X_train)
X_test = s.fit_transform(X_test)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train=s.fit_transform(X_train)
# X_test=s.fit_transform(X_test)

"""##Task2
# 5-Random Forest
"""

classifier5 = RandomForestClassifier()

classifier5.fit(X_train, Y_train)

# pickle.dump(classifier5, open("model.pkl", "wb"))
with open('model.pkl', 'wb') as f:
    pickle.dump((s, classifier5), f)


y_predict = classifier5.predict(X_test)
