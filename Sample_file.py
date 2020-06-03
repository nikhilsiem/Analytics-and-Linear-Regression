
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Importing the dataset
df=pd.read_csv("C:\\Users\\Administrator\\Desktop\\Data analytics\\Teacher_salary_csv.csv")
df.head() #check the first five rows of Dataset
d=df.iloc[:,-2]

#Outliers using IQR method
#calculating outliers without replacing NaN value otherwise we can replace the Nan value with 0 using fillna() hence the output  result changes

sorted(d)
d.describe()
d.dropna(inplace=True)
q1, q3= np.percentile(d,[25,75])
print(q1,q3)
iqr = q3 - q1
print(iqr)
lower_bound = q1 -(1.5 * iqr) 
upper_bound = q3 +(1.5 * iqr)
outlier_qmr=[]
for i in d:
    if(i<lower_bound or i>upper_bound):
        outlier_qmr.append(i)
print(outlier_qmr)

#Outliers using Z-score method
#calculating outliers without replacing NaN value otherwise we can replace the Nan value with 0 using fillna() hence the output  result changes
outliers_z=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers_z.append(y)
    return outliers_z
 print(outliers_z)


# Working with categorical data
obj_df = df.select_dtypes(include=['object']).copy()
obj_df.head()
obj_df["Marital Status"].value_counts()
obj_df["Gender"].value_counts()
obj_df["Campus"].value_counts()
obj_df["Highest Degree"].value_counts()
#changing categorical data into numerical continous data according to the frequency
prepross = {"Marital Status":     {"M": 1, "U": 0},
                "Gender": {"M": 1, "F":0 },
               "Campus": {"Goa":2,"Hyderabad":1,"Pilani":0},
               "Highest Degree": {"PhD.":2,"Graduation":1,"Post graduation":0}}
obj_df.replace(prepross, inplace=True)
obj_df.head()

# Taking care of the Missing data
df.fillna(df.median(), inplace=True)#using median rather than men so that it does not get affected by outliers



# Feature Scaling
scaler = preprocessing.StandardScaler() #can also use MinMaxScaler but using Standardisation instead for more generic results
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=['Marital Status', 'Gender', 'Campus','Age','Highest Degree','years of teachingexperience','Number of research publications','Projects completed','Ph.D students guided','Average feedback','Yearly Salary'])


# Splitting into training and testing

X=df.iloc[:,5:-1]#all columns starting from age till second last column
y=df[:,12]#yearly income as output
#Create dummy variale value for categorical data
mar=pd.get_dummies(X["Marital Status"],drop_first=True)
gen=pd.get_dummies(X["Gender"],drop_first=True)
cam=pd.get_dummies(X["Campus"],drop_first=True)
deg=pd.get_dummies(X["Highest Degree"],drop_first=True)
#removing categorical data columns
X=X.drop("Marital Status",axis=1)
X=X.drop("Gender",axis=1)
X=X.drop("Campus",axis=1)
X=X.drop("Highest Degree",axis=1)
#adding new dummy value columns
X=pd.concat([X,mar],axis=1)
X=pd.concat([X,gen],axis=1)
X=pd.concat([X,cam],axis=1)
X=pd.concat([X,deg],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
print X_train.shape, y_train.shape
print X_test.shape, y_test.shape


# Training the Multiple Linear Regression model on the Training set
reg=LinearRegression()
reg.fit(X_train,y_train)

# Predicting the Test set results
y_pred= reg.predict(X_test)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)#for crosschecking predicted value

# (Optional) graphical representation
plt.scatter(X_test,y_test, label='Multiple Linear Regression')
plt.xlabel('Multiple Variables')
plt.ylabel('Yearly Income')
plt.title('Scatter plot for Dependent Variables vs Income')
plt.legend()
plt.show()