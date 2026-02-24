###########################################
# Lineer Regresyon ile satış tahmini
###########################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.float_format', lambda x: '%.2f' %x)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_score
df=pd.read_csv("advertising.csv")
df.shape
x= df[["TV"]]
y=df[["Sales"]]
##############################
# Model
##############################
reg_model = LinearRegression().fit(x, y)

# y_hat=a+w*x
# sabit (b-bias)
reg_model.intercept_[0]
#Tv'nin katsayısı (w1)
reg_model.coef_[0][0]

##############################
# Tahmin
##############################

# 150 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0]+reg_model.coef_[0][0]*150

# 500 birimlik TV harcaması olsa ne kadar satış olması beklenir?
reg_model.intercept_[0]+reg_model.coef_[0][0]*500
df. describe().T

# Modelin Görselleştirilmesi
g=sns.regplot(x="TV", y="Sales", data=df, scatter_kws={'color':'b', 's': 9},
              ci=False, color="r")
g.set_title(f"Model Denklemi: Sales={round(reg_model.intercept_[0], 2)}+Tv*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom=0)
plt.show()