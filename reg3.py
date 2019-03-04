print(__doc__)


# Code source: Jaques Grobler
# License: BSD 3 clause

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd 
import statsmodels.formula.api as smf
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 
data1 = pd.read_csv("Book1.csv",delimiter=';') 
# Preview the first 5 lines of the loaded data 

print(data1[:5])
lndata1=np.log(data1)
print(lndata1[:5])
#Let's see descriptive statistics 
print(data1.describe())
print(lndata1.describe())
# Load the diabetes dataset
#plot 
plt.figure(1)
plt.scatter(data1.speed, data1.dist,  color='black')


plt.figure(2)
plt.scatter(lndata1.speed, lndata1.dist,  color='red')




df = pd.DataFrame(data1)
print(df)
results = smf.ols('speed ~ dist', data=df).fit()

print(results.summary())

# Plot outputs
#plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
#plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.show()
