# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 

# code starts here
data = pd.read_csv(path)
bank = pd.DataFrame(data)

#check all categorical values.
categorical_var = bank.select_dtypes(include = 'object')
print('categorical_var:',categorical_var)

#check all categorical values.
numerical_var = bank.select_dtypes(include = 'number')
print('numerical_var:',numerical_var)

# code ends here


# --------------
# code starts here
bank.drop('Loan_ID', axis = 1, inplace = True)
banks = bank
print('Null values :\n',banks.isnull().sum())
bank_mode = banks.mode().iloc[0]
banks['Gender'].fillna(bank_mode['Gender'], inplace = True)
banks['Married'].fillna(bank_mode['Married'], inplace = True)
banks['Dependents'].fillna(bank_mode['Dependents'], inplace = True)
banks['Self_Employed'].fillna(bank_mode['Self_Employed'], inplace = True)
banks['LoanAmount'].fillna(bank_mode['LoanAmount'], inplace = True)
banks['Loan_Amount_Term'].fillna(bank_mode['Loan_Amount_Term'], inplace = True)
banks['Credit_History'].fillna(bank_mode['Credit_History'], inplace = True)
print('Banks :\n',banks)
#code ends here


# --------------
# Code starts here
avg_loan_amount = pd.pivot_table(banks,index=['Gender', 'Married', 'Self_Employed'],values=['LoanAmount'],aggfunc=np.mean)
print(avg_loan_amount)


# code ends here



# --------------
# code starts here
con1 = banks['Self_Employed']=='Yes'
con2 = banks['Loan_Status']=='Y'
loan_approved_se = banks[con1 & con2]

con3 = banks['Self_Employed']=='No'
con4 = banks['Loan_Status']=='Y'
loan_approved_nse = banks[con3 & con4]

Loan_Status = 614
percentage_se = (len(loan_approved_se)/Loan_Status)*100
percentage_nse = (len(loan_approved_nse)/Loan_Status)*100
print('percentage_se:',percentage_se, '\npercentage_nse:',percentage_nse)
# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x:x/12)

con = loan_term >=25
big_loan_term = len(banks[con])
print('big_loan_term:',big_loan_term)


# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')['ApplicantIncome', 'Credit_History']

mean_values = loan_groupby.mean()
print('mean_values:',mean_values)

# code ends here


