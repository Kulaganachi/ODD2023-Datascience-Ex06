# Feature_Transformation
## Ex-06-Feature-Transformation
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
### STEP 1:
Read the given Data

### STEP 2:
Clean the Data Set using Data Cleaning Process

### STEP 3:
Apply Feature Transformation techniques to all the features of the data set

### STEP 4:
Print the transformed features

## PROGRAM:
 NAME:Kulaganachi
 
 REG NO: 212221040086
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.HighlyNegativeSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModeratePositiveSkew,fit=True,line='45')
plt.show()

sm.qqplot(df.ModerateNegativeSkew,fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT:
![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/2b72f684-63ad-41dc-8b8c-fc1d9b0a6763)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/25c638fa-6432-4c89-9d74-786a425b5b7a)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/d376935e-7111-437b-b417-1c18e969773f)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/c0c1215f-c14a-469b-b442-f7077c0849cf)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/2f479624-1c3e-4c58-8aa3-46b95d96a0e5)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/18c1ca12-b58f-421c-9e1e-b17fd5636dad)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/4ebb9d04-4152-4476-929e-e935aa9fb257)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/5bc0e1cd-148b-413b-b300-c201a1c371be)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/f09f3490-9dac-4a86-a686-5a2f206a14e9)


![image](https://github.com/Kulaganachi/Feature_Transformation/assets/133641126/96f7f95e-80b7-47ac-82fe-b944bad7ea40)


## RESULT:
Thus feature transformation is done for the given dataset.

