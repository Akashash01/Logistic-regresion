#IMPORT LIBRARY
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import os
import statsmodels.api as sm
import pylab as pl

#LOAD FILES
os.chdir('C:\\Users\\NEW\\Desktop\\python\\datasets')
data = ash.copy()
# import dataset
data = pd.read_csv('data.csv')
pd.set_option('display.max_columns', None)#show all columns

#WIEGHT OF EVIDENCE AND INFORMATION VALUE(to segregate independent variable)
def iv_woe(data, target, bins=10, show_woe=False):
    
    #Empty Dataframe
    newDF,woeDF = pd.DataFrame(), pd.DataFrame()
    
    #Extract Column Names
    cols = data.columns
    
    #Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars], bins,  duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(),6)))
        temp =pd.DataFrame({"Variable" : [ivars], "IV" : [d['IV'].sum()]}, columns = ["Variable", "IV"])
        newDF=pd.concat([newDF,temp], axis=0)
        woeDF=pd.concat([woeDF,d], axis=0)

        #Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF

## calling the function
# data is the name of the dataset
# Churn is the dependent varaible
iv, woe = iv_woe(data = data, target = 'Is_Lead', bins=10, show_woe = True)
print(iv)
print(woe)
iv = pd.DataFrame(iv)
iv.sort_values(["IV"],ascending=[0])

#HINT
According to Siddiqi (2006), by convention the values of the IV statistic can be interpreted as follows. If the IV statistic is:
•Less than 0.02, then the predictor is not useful for modeling (separating the Goods from the Bads)
•0.02 to 0.1, then the predictor has only a weak relationship to the Goods/Bads odds ratio
•0.1 to 0.3, then the predictor has a medium strength relationship to the Goods/Bads odds ratio
•0.3 or higher, then the predictor has a strong relationship to the Goods/Bads odds ratio.

#DATA WRANGLING(preprocessing)
data = data.drop(["ID","Region_Code","Credit_Product"], axis  = 1)
data

#removing outlier
def outlier (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data
data.boxplot("Avg_Account_Balance")
data = outlier(data,"Avg_Account_Balance")
data.dtypes

#remove unwanted space(if required)
dataset["Credit_Product"] =dataset["Credit_Product"].replace(r'\s+',np.nan,regex = True)

#missing values
data.isnull().sum()#show is any variable has any missing value
data.dropna() or fillna(0 or 1) to balance data

#replace str to int(if required)
data["Credit_Product"] = data["Credit_Product"].replace(["NaN","Yes","No"],[0,1,0])

#dummy variables
ak = data.loc[:,["Gender","Occupation","Channel_Code"]]
ak
data = data.drop(["Gender","Occupation","Channel_Code"],axis = 1)
dum = pd.get_dummies(ak.astype(str),drop_first=True)
dum

#concatenation
data = pd.concat([data,dum],axis=1)
data

#LOGISTIC REGRESSION MODEL
train_cols = data.loc[:,["INDEPENDENT WITH DUMMY VARIABLES "]]
logit = sm.Logit(data['DEPENDENT VARIABLE'], train_cols)
rock = logit.fit()
rock.summary()#all p values are significant

#INTERPRETATION
var = pd.DataFrame(round(rock.pvalues,3))# shows p value
var["coeff"] = rock.params#coefficients
#rename columns
var.columns.values[[0,1]]= ["p value","coefficients"]
var
cov = rock.cov_params()
std_err = np.sqrt(np.diag(cov))
var["z"]=rock.params.values/std_err
var
rock.conf_int()# confidence interval
np.exp(rock.params)


#LINEAR MODEL FOR VIF CALCULATION
import numpy as np
import pandas as pd
import statsmodels.formula.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#MODEL
result=sm.ols(formula="Is_Lead~Region_Code+Credit_Product+Avg_Account_Balance+Is_Active+Q('Channel_Code_X4')",
             data=data).fit()
result.summary()# shows total summary

#CHECK MULTICOLLINEARITY USING VIF
#remove variable based on vif
#all vif values are under 2, hence no variable is removed

var = pd.DataFrame(round(result.pvalues,3))# shows p value
var["coeff"] = result.params#coefficients
variables = result.model.exog #.if I had saved data as rock
# this it would have looked like rock.model.exog
vif = [variance_inflation_factor(variables, i) for i in range(variables.shape[1])]
vif 
var["vif"] = vif
var

#AGAIN LOGISTIC MODEL WITH SEGREGATED VARIABLE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
inputData=train_cols # ind var
outputData=data.loc[:,"Is_Lead"] 
logit1=LogisticRegression()
logit1.fit(inputData,outputData)
logit1.score(inputData,outputData)

#PREDICTION
y_pred = logit1.predict(train_cols)
prob = logit1.predict_proba(train_cols)
#prob.count()
#transform the probabilities into DataFrame
# it shows probability for both
prob = pd.DataFrame(prob)
prob = prob.iloc[:,1]#showing the probability of being 1
prob = prob.reset_index()



#RESULTS AND INTERPRETATION
outputData = pd.DataFrame(outputData)
outputData.head()
outputData = outputData.reset_index()
outputData = outputData.iloc[:,1]



rock = pd.concat([outputData,prob], axis=1)
rock = rock.iloc[:,[0,2]]#this line might give error, check the column index

df = rock.copy()
df.columns = ["y","p"]

#AUC 
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)# this is experimental not required

##Computing false and true positive rates
fpr, tpr,_=roc_curve(logit1.predict(inputData),outputData,drop_intermediate=False)
import matplotlib.pyplot as plt
plt.figure()
##Adding the ROC
plt.plot(fpr, tpr, color='red',
 lw=2, label='ROC curve')
##Random FPR and TPR
plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
##Title and label
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
roc_auc_score(logit1.predict(inputData),outputData)



#KS-STAT (lies between 0-1)
df = rock.copy()
df.columns = ["y","p"]
df.head()

new = df.copy()
new.columns = ["Is_Lead","Prob"]
new.head()

new['decile'] = pd.qcut(new['Prob'],10,
labels=['1','2','3','4','5','6','7','8','9','10'])
new.head()

new.columns = ['Defaulter','Probability','Decile']
new.head()
new['Non-Defaulter'] = 1-new['Defaulter']
new.head()


boogieman = pd.pivot_table(data=new,index=['Decile'],
values=['Defaulter','Non-Defaulter','Probability'],
aggfunc={'Defaulter':[np.sum],'Non-Defaulter':[np.sum],
'Probability' : [np.min,np.max]})
boogieman.head()
boogieman.reset_index()

boogieman.columns = ['Defaulter_Count','Non-Defaulter_Count','max_score','min_score']
boogieman['Total_Cust'] = boogieman['Defaulter_Count']+boogieman['Non-Defaulter_Count']
boogieman

kane = boogieman.sort_values(by='min_score',ascending=False)
kane


kane['Default_Rate'] = (kane['Defaulter_Count'] / 
kane['Total_Cust']).apply('{0:.2%}'.format)
default_sum = kane['Defaulter_Count'].sum()
non_default_sum = kane['Non-Defaulter_Count'].sum()
kane['Default %'] = (kane['Defaulter_Count']/
default_sum).apply('{0:.2%}'.format)
kane['Non_Default %'] = (kane['Non-Defaulter_Count']/
non_default_sum).apply('{0:.2%}'.format)
kane

kane['ks_stats'] = np.round(((kane['Defaulter_Count'] / 
kane['Defaulter_Count'].sum()).cumsum() -
(kane['Non-Defaulter_Count'] / 
kane['Non-Defaulter_Count'].sum()).cumsum()), 4) * 100
kane

flag = lambda x: '*****' if x == kane['ks_stats'].max() else ''
kane['max_ks'] = kane['ks_stats'].apply(flag)
kane
