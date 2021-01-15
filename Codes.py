#Load Package's That I Will Use

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.formula.api as sma
import statsmodels.stats.diagnostic as ds
import statsmodels.api as sa
from statsmodels.tsa.stattools import adfuller, acf, pacf, kpss

#Load Data and Clean It

cdf = pd.read_csv(r'c:\users\ag\desktop\sudan\money.csv')
cdf = cdf.drop(columns = 'Unnamed: 0')
cdf2 = pd.read_csv(r'c:\users\ag\desktop\sudan\inflation.csv')
cdf2 = cdf2.drop(columns = 'Unnamed: 0')

# Check for na and null

cdf.isna()
cdf.isnull()

cdf2.isna()
cdf2.isnull()


# Connecte Both Dataframes

data = pd.concat((cdf, cdf2), 1)


#Correlation Matrix for Multicollinearty

data.corr()

#drop data that we will not use

DataUse = data.drop(columns = ['MB', 'local'])


# Chack for Stationary

# Soerate The Data
 
inf = DataUse[['inflation']]
 
fore = DataUse[['foreign']]

ms = DataUse[['MS']] 

#Inflation

urt = adfuller(inf)
print("ADF Statistics %f" % urt[0])
print("p-value %f" % urt[1])
print("Critical Value:")
for key, value in urt[4].items():
    print('\t%s: %.3f' %(key, value))
    
if urt[0] < urt[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time series is non-stationary")

inf = pd.DataFrame.diff(inf, 1).dropna()

#Foreign Currency

urt1 = adfuller(fore)
print("ADF Statistics %f" % urt1[0])
print("p-value %f" % urt1[1])
print("Critical Value:")
for key, value in urt1[4].items():
    print('\t%s: %.3f' %(key, value))
    
if urt1[0] < urt1[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time series is non-stationary")


fore = pd.DataFrame.diff(fore, 1).dropna()

#Money Supply

urt2 = adfuller(work.MS)
print("ADF Statistics %f" % urt2[0])
print("p-value %f" % urt2[1])
print("Critical Value:")
for key, value in urt2[4].items():
    print('\t%s: %.3f' %(key, value))
    
if urt2[0] < urt2[4]["5%"]:
    print("Reject Ho - Time Series is Stationary")
else:
    print("Failed to Reject Ho - Time series is non-stationary")

ms = pd.DataFrame.diff(ms, 1).dropna()

#Create Work DataFrame

work = pd.concat((inf[2:13], fore[2:13], ms), 1)

#Correlation Matrix for Multicollinearty

work.corr()

#Visualization the data

plt.subplots(3, 1, sharex=(True))
plt.subplot(311)
plt.plot(DataUse.MS, color = 'g', label = 'Money Supply')
plt.legend(loc = 'best')
plt.subplot(312)
plt.plot(DataUse.foreign, color = 'r', label = 'Foeiegn Currency')
plt.legend(loc = 'best')
plt.subplot(313)
plt.plot(DataUse.inflation, color = 'b', label = 'Inflation')
plt.legend(loc = 'best')
plt.show()

#Basic Statistics

#Inflation
meaninf = np.mean(DataUse.inflation)
sdinf = np.std(DataUse.inflation)
varinf = np.var(DataUse.inflation)

#Foreign Currency

meanfor = np.mean(DataUse.foreign)
sdfor = np.std(DataUse.foreign)
varfor = np.var(DataUse.foreign)

#Money Supply

meanMS = np.mean(DataUse.MS)
sdMS = np.std(DataUse.MS)
varMS = np.var(DataUse.MS)


#Create Model

Model = sma.wls('work.inflation ~ foreign + MS', work).fit()

print(Model.summary())

#Weighted Least Squares used to fix Heteroscedastisity

#Test The Model

Heteroscedastisity = ds.het_white(Model.resid, exog = work)

print('F-statistic %r' % Heteroscedastisity[2])
print('Prob,F %f' % Heteroscedastisity[3])
print('Chi-Square %s' % Heteroscedastisity[0])
print('Prob,Chi-Square %g' % Heteroscedastisity[1])

Autocorrelation = ds.acorr_breusch_godfrey(Model, nlags=(2))

print('F-statistic %r' % Autocorrelation[2])
print('Prob,F %f' % Autocorrelation[3])
print('Chi-Square %s' % Autocorrelation[0])
print('Prob,Chi-Square %g' % Autocorrelation[1])
