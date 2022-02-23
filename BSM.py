from math import log, nan, sqrt, pi, exp
from numpy.core.numeric import NaN
from scipy.stats import norm #normal distribution for probabilities
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import linregress

#D1 and D2 calculations
def BSd1(A,L,T,r,v): #given in the book
    return(log(A/L)+(r+ 0.5*v**2)*T)/(v*(T**0.5))
def d2(A,L,T,r,v): #given in the book
    return BSd1(A,L,T,r,v)-v*sqrt(T)

#BSM MODEL
#price option
def BSM(E,A,L,T,r,v):
    p = (E+L*exp(-r*T)*norm.cdf(d2(A,L,T,r,v)))/norm.cdf(d2(A,L,T,r,v))
    return p


## Data Extraction from Final Data File##
xls = pd.ExcelFile('Final Data File.xlsx')
company_data = pd.read_excel(xls,'Fahad-Data')
market_value_data = pd.read_excel(xls,'MV Daily').iloc[1: , :]
total_liability_data = pd.read_excel(xls,'Current Liabilities Yearly').iloc[1: , :]
risk_free_rate_data = pd.read_excel(xls,'rf daily')
## Data Extraction from KSE-100##
xls = pd.read_csv('KSE-100_03012011_01102021.csv')
s_p_date = xls['Date']
s_p_index = (xls['Close'] - xls['Open']) / xls['Open']

for i in range (1,670):
    try:
        market_value_data = market_value_data.drop("#ERROR."+str(i), axis = 1)
        total_liability_data = total_liability_data.drop("#ERROR."+str(i), axis = 1)
    except:
        continue

year =[2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021]

## testing for single company
market_date = market_value_data. iloc[:, 0]
liability_year = total_liability_data. iloc[:, 0]
rf_date = risk_free_rate_data. iloc[:, 0]

result ={}
for c in range(1,6):
    np.warnings.filterwarnings('ignore')
    result[list(market_value_data.columns)[c][:-14]] = []
    market_value = market_value_data. iloc[:, c]
    liability = total_liability_data. iloc[:, c]
    rf = risk_free_rate_data. iloc[:,1]


    for y in year:
        asset_iter_k = {}
        asset_iter_k_1 = {}
        ## table 2.1 of the book
        for i in range(len(market_date)):
            for j in range(len(liability_year)):
                if liability_year.iloc[j] == market_date.iloc[i].year == y:
                    asset_iter_k[market_date.iloc[i]] = liability.iloc[j] + market_value.iloc[i]

        in_returns = {}
        for i in range(1,len(asset_iter_k)):
            in_returns[list(asset_iter_k.keys())[i]] = np.log(asset_iter_k[list(asset_iter_k.keys())[i]]/asset_iter_k[list(asset_iter_k.keys())[i-1]])*100

        asset_volatility = np.std(pd.DataFrame(in_returns.values()))*(260**0.5)*100

        ##print("Asset Volatility: ", asset_volatility)

        """ for i in range(len(market_date)):
            for j in range(len(liability_year)):
                if liability_year.iloc[j] == market_date.iloc[i].year == y:
                    asset_iter_k_1[market_date.iloc[i]] = BSM(market_value.iloc[i],asset_iter_k[list(asset_iter_k.keys())[i]],liability.iloc[j],1,rf.iloc[i],asset_volatility)

        MSE = np.square(np.subtract(list(asset_iter_k.values()),list(asset_iter_k_1.values()))).mean() """


        ## table 2.2

        ## Excess Return 
        for i in range(len(market_date)):
            for j in range(len(liability_year)):
                if liability_year.iloc[j] == market_date.iloc[i].year == y:
                    rf_value = rf.iloc[i]
                    break

        asset_value = {}
        for i in range(1,len(asset_iter_k)):
            asset_value[list(asset_iter_k.keys())[i]] = ((asset_iter_k[list(asset_iter_k.keys())[i]]/asset_iter_k[list(asset_iter_k.keys())[i-1]])-(1+(rf_value)/260))*100

        s_p = {}
        for i in range(1, len(s_p_index)):
            if int(s_p_date.iloc[i][-4:]) == int(y):
                value = ((s_p_index.iloc[i]/s_p_index.iloc[i-1])-(1+(rf_value)/260))*100
                s_p[s_p_date.iloc[i]] = value

        asset_value = pd.Series(asset_value).to_frame().dropna()
        try:
            slope, intercept, r, p, se = linregress(list(s_p.values()), asset_value[0].iloc[:len(list(s_p.values()))])
        except:
            slope = 0
        drift_rate = np.log(1+slope)*100
        ##print("Drift rate is:", drift_rate)

        ## Table 2.3
        A_t = list(asset_iter_k.values())[-1]
        for j in range(len(liability_year)):
            if liability_year.iloc[j] == y:
                L_t = liability.iloc[j]
                break

        distance_to_default = (np.log(A_t)+(drift_rate/100-(asset_volatility/100)**2/2)-np.log(L_t))/(asset_volatility/100)
        ##print("Distance to default is: ", distance_to_default)
        default_probability = list(norm.cdf(-1*distance_to_default))
        result[list(market_value_data.columns)[c][:-14]].append(default_probability[0]*100)
        ##print("Deafualt Probability is: ", default_probability[0]*100)


result = pd.DataFrame(result, index= year)
print(result)