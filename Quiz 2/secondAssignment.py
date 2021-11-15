import pandas as pd
quiz_data = pd.read_csv('./quiz_data.csv')

freq = pd.crosstab("Yes", quiz_data.Insurance, normalize='index')
print(freq) #GINI of total data is 0.5

#SEX
absfreq = pd.crosstab(quiz_data.Sex, quiz_data.Insurance)
freq = pd.crosstab(quiz_data.Sex, quiz_data.Insurance, normalize='index')
GINI_Sex_Male = 1 - freq.loc["M", "No"]**2 - freq.loc["M", "Yes"]**2
#print(GINI_Sex_Male)
GINI_Sex_Female = 1 - freq.loc["F", "No"]**2 - freq.loc["F", "Yes"]**2
freqSum = pd.crosstab(quiz_data.Sex, quiz_data.Insurance, normalize='all').sum(axis=1)

GINI_Sex = freqSum.loc["F"]*GINI_Sex_Female + freqSum.loc["M"]*GINI_Sex_Male
print(GINI_Sex)

#CarType
freq = pd.crosstab(quiz_data.CarType, quiz_data.Insurance, normalize='index')
#print(freq)
GINI_Car_Family = 1 - freq.loc["Family", "No"]**2 - freq.loc["Family", "Yes"]**2
GINI_Car_Sedan = 1 - freq.loc["Sedan", "No"]**2 - freq.loc["Sedan", "Yes"]**2
GINI_Car_Sport = 1 - freq.loc["Sport", "No"]**2 - freq.loc["Sport", "Yes"]**2
freqSum = pd.crosstab(quiz_data.CarType, quiz_data.Insurance, normalize='all').sum(axis=1)
print(freqSum)

GINI_Car = freqSum.loc["Family"]*GINI_Car_Family + freqSum.loc["Sedan"]*GINI_Car_Sedan + freqSum.loc["Sport"]*GINI_Car_Sport
print(GINI_Car)

#Budget
freq = pd.crosstab(quiz_data.Budget, quiz_data.Insurance, normalize='index')
#print(freq)
GINI_Budget_Low = 1 - freq.loc["Low", "No"]**2 - freq.loc["Low", "Yes"]**2
GINI_Budget_Medium = 1 - freq.loc["Medium", "No"]**2 - freq.loc["Medium", "Yes"]**2
GINI_Budget_High = 1 - freq.loc["High", "No"]**2 - freq.loc["High", "Yes"]**2
GINI_Budget_VeryHigh = 1 - freq.loc["VeryHigh", "No"]**2 - freq.loc["VeryHigh", "Yes"]**2
freqSum = pd.crosstab(quiz_data.Budget, quiz_data.Insurance, normalize='all').sum(axis=1)
print(freqSum)

GINI_Budget = freqSum.loc["Low"]*GINI_Budget_Low + freqSum.loc["Medium"]*GINI_Budget_Medium + freqSum.loc["High"]*GINI_Budget_High + freqSum.loc["VeryHigh"]*GINI_Budget_VeryHigh
print(GINI_Budget)

