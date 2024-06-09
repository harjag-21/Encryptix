import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("Churn_Modelling.csv")
data.head()

data = data.drop(['RowNumber', 'CustomerId', 'Surname'],axis=1)


plt.figure(figsize =(8,6))
sns.countplot(x='Exited',data = data)