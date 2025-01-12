
import pandas as pd 
import numpy as np 


#Importe la base des données fondamentaux
fund = pd.read_excel('Fun_data.xlsx')
#on doit nettoyer les colonnes de chiffre et ensuite le convertire en float
fund[2021] = fund[2021].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2022] = fund[2022].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2023] = fund[2023].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
#Corriger le nom de quelque Tickers
fund['Ticker'].iloc[0] = 'AFM'
fund['Ticker'].iloc[144] = 'DIS'
fund['Ticker'].iloc[258] = 'MNG'
#Enlève les valeurs manquantes dans la colonne Ticker
fund['Ticker'] = fund['Ticker'].fillna(method='ffill')

#Rérganiser notre base de donnée pour que cette fois ci chaque colonne devient uen variable et chaque ligne une observation
df_long = fund.melt(id_vars=["Ticker", "Ratio"], var_name="Year", value_name="Value")

# Etape 2  : On pivote notre base 
df_tidy = df_long.pivot_table(index=["Ticker", "Year"], columns="Ratio", values="Value").reset_index()

# Etape 3 : on double indexe notre base de donnée par l'année et le Ticker
df_tidy = df_tidy.set_index(["Ticker", "Year"]).sort_index()
#On élimine la colonne PA car elle n'est présente que pour quelque actions
df_tidy = df_tidy.drop(columns = {'PA'})

from sklearn.linear_model import LinearRegression
import pandas_ta as ta
from arch import arch_model

#Pour calculer la tangente qu'on va utiliser par lasuite pour la création du facteur Price_acc
def slope(price):
    if len(price) < 2:
        return 0
    X = np.arange(len(price)).reshape(-1,1)
    y = price.values.reshape(-1,1)
    model = LinearRegression().fit(X,y)
    return model.coef_[0][0]

def factor_data(data):
    #Momentum factors
    Acc = []
    for i in range(126, len(data['Close'])):
        long = data['Close'].iloc[i-126:i]
        short = data['Close'].iloc[i-21:i] 
        l_slop = slope(long)
        s_slop = slope(short)
        vol = np.std(long)
        a = (l_slop-s_slop)/vol if vol!=0 else 0
        Acc.append(a)
    Acc = [np.nan]*126+Acc
    Acc = Acc[:len(data)]
    #Price accelerator 
    data['Price_acc'] = Acc
    #RSI
    data['RSI'] = ta.rsi(data['Close'], length=14)
    #Price momentum
    data['Price_mom'] = data['Return'].rolling(window=252, min_periods=1).sum().shift(21) 
    # Le prix high du dernier 53 semaines
    data['52_week high'] = ((data['Close'] - data['Close'].rolling(window = 252, min_periods = 1).max())/data['Close'].rolling(window = 252, min_periods = 1).max())*100
    #Bollinger bands 
    data.ta.bbands(close = data['Close'], length=20, std = 2, append = True)
    #Volatility glissante
    data['Vol'] = data['Return'].rolling(window = 25).std()/np.sqrt(252)
    #Volatilité implicite
    model_a = arch_model(data['Return'].iloc[1:], vol='Garch', p=1, q=1)
    garch_fit = model_a.fit()
    b = garch_fit.conditional_volatility
    b = pd.concat([pd.Series([np.nan]), b]).reset_index(drop = True)
    data['garch_vol'] = b
    return data

def data_prep(tick):
    #Importe la base de donnée du ticker
    data = pd.read_excel(f'Stocks//{tick}.xlsx')
    #Organiser les coonnes et leurs types
    data['Séance'] = pd.to_datetime(data['Séance'], format = '%d/%m/%Y')
    data = data[['Séance', 'Ticker', 'Dernier Cours', 'Nombre de titres échangés', 'Capitalisation' ]]
    data = data.rename(columns = { 'Dernier Cours': 'Close', 'Nombre de titres échangés': 'Volume', 'Capitalisation': 'Market Cap'})
    #Calcule les rendements passé de 1, 3 et 6 mois
    data['Return'] = data['Close'].pct_change()
    data['R1M'] = data['Close'].pct_change(21)
    data['R3M'] = data['Close'].pct_change(63)
    data['R6M'] = data['Close'].pct_change(126)
    #On ajoute les facteur de la fonction factor_data
    data = factor_data(data)
    #Initialiser les facteurs fondamentaux
    data['BPA'] = np.nan
    data['Div_yield'] = np.nan
    data['ROE'] = np.nan
    data['PER'] = np.nan
    data['Payout'] = np.nan
    for i in range(len(data)):
        #Pour l'année 2024 vu le manque de donnée on va utliser les ratio de 2023 qu'on va fetcher de la base fund
        if data['Séance'].dt.year.iloc[i] == 2024:
            data['BPA'].iloc[i] = df_tidy['BPA'].loc[(tick, 2023)]
            data['Div_yield'].iloc[i] = df_tidy['Dividend yield (en %)'].loc[(tick, 2023)]
            data['ROE'].iloc[i] = df_tidy['ROE (en %)'].loc[(tick, 2023)]
            data['PER'].iloc[i] = df_tidy['PER'].loc[(tick, 2023)]
            data['Payout'].iloc[i] = df_tidy['Payout (en %)'].loc[(tick, 2023)]
        else : 
            data['BPA'].iloc[i] = df_tidy['BPA'].loc[(tick, data['Séance'].dt.year.iloc[i])]
            data['Div_yield'].iloc[i] = df_tidy['Dividend yield (en %)'].loc[(tick, data['Séance'].dt.year.iloc[i])]
            data['ROE'].iloc[i] = df_tidy['ROE (en %)'].loc[(tick, data['Séance'].dt.year.iloc[i])]
            data['PER'].iloc[i] = df_tidy['PER'].loc[(tick, data['Séance'].dt.year.iloc[i])]
            data['Payout'].iloc[i] = df_tidy['Payout (en %)'].loc[(tick, data['Séance'].dt.year.iloc[i])]
    return data

#Import la liste des actions qui constituent l'indice MASI 20
idx = pd.read_excel('composition-history-1734947403.xlsx')
Tck = idx['Ticker']

#Création de la base de donée finale avec tous les actifs et leurs facteurs
merged_df = pd.DataFrame()
for tick in Tck:
    merged_df = pd.concat([merged_df, data_prep(tick)],ignore_index = True)
    



#Oragnise la base 
merged_df = merged_df.rename(columns={'Séance' : 'Date'})
#Double indexe la base par la date et le ticker
merged_df = merged_df.set_index(['Date', 'Ticker'])

#Création de Trois nouvelle variables qui sont le rendement future de 1, 3 et 6 mois et ceci pour chaque ticker
for t in [1,3,6]:
    merged_df[f'target_{t}m'] = (merged_df.groupby(level='Ticker')[f'R{t}M'].shift(-t))


#Elimine l'indexe 
merged_df = merged_df.reset_index()

merged_df

#Ajoute les variables de mois et d'année 
merged_df['Month'] = merged_df['Date'].dt.month
merged_df['Year'] = merged_df['Date'].dt.year



import seaborn as sns
import matplotlib.pyplot as plt
#Tarcer la relation entre les facteurs les plus important et la variable target_1m
cols = ['Market Cap', 'Price_acc', 'Vol', 'ROE', 'garch_vol','Div_yield', 'target_1m', 'Date']
data_corr = merged_df[cols]
data_corr = data_corr.groupby('Date').corr()[['target_1m']].reset_index()

data_corr=data_corr.loc[data_corr[data_corr.level_1 !="target_1m"].index]
data_corr.rename(columns = {'level_1' : 'Factors'}, inplace = True)
data_corr
plt.figure(figsize = (12,6))

sns.swarmplot(x = 'Factors', y = 'target_1m', data = data_corr, palette='Set1')
plt.show()
plt.savefig('Boxplot.png')

import seaborn as sns

import matplotlib.pyplot as plt

# Calcule la matrice de corrélation
corr_matrix = merged_df.drop(columns = ['Date', 'Month', 'Year', 'Ticker']).corr()

# Visualization de la matrice
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()
plt.savefig('Corr_matrix.png')

#Création de la base qui contient que les rendements
returns = merged_df.pivot(index='Date', columns='Ticker', values='Return')



returns_3 = pd.read_csv('Ret_mon.csv')
returns_3 = returns_3.drop(columns = {'Unnamed: 0'})

from sklearn.linear_model import ElasticNet

nb_port = 3
T = len(returns_3)
port_weight = {}
port_returns = {}
def weights_sparse(returns, alpha, Lambda):
    weights = []
    lr = ElasticNet(alpha = alpha, l1_ratio = Lambda)
    for tick in returns_3.columns:
        y = returns[tick].values
        X = returns.drop(tick, axis = 1).values
        lr.fit(X, y)
        err = y - lr.predict(X)
        w = (1-np.sum(lr.coef_))/np.var(err)
        weights.append(w)
    return weights/np.sum(weights)

def weights_multi(returns, j, alpha, Lambda):
    N = returns.shape[1]
    if j == 0:
        return np.repeat(1/N,N)
    elif j ==1:
        sigma = np.cov(returns.T)+10**(-2)*np.identity(N)
        w = np.matmul(np.linalg.inv(sigma), np.repeat(1, N))    
        return w/np.sum(w)
    elif j ==2:
        return weights_sparse(returns, alpha, Lambda)
    
        

T = list(Tck)
a =  [ col for col in returns_3.columns if   (col in Tck)]



returns_3.columns = returns_3.columns.str.replace('_month', '')
returns_3 = returns_3[T]
returns_3.head()




for m, month in np.ndenumerate(returns.index):
    temp_data = returns
    realised_returns = returns.loc[returns.index == month].values
    weights_temp = {}
    returns_temp = {}
    for j in range(nb_port):
        wgts = weights_multi(temp_data, j, 0.1,0.1)
        rets = np.sum(wgts*realised_returns)
        weights_temp[j] = wgts
        returns_temp[j] = rets
    port_weight[month] = weights_temp
    port_returns[month] = returns_temp
    
port_returns_final = pd.concat(
    {k: pd.DataFrame.from_dict(v, 'index') for k, v in port_returns.items()},
    axis = 0
).reset_index()

colnames = ['date', 'strategy', 'return']
port_returns_final.columns = colnames
strategies_name = {0:'EW', 1:'MV', 2:'Sparse'}
port_returns_final['strategy'] = port_returns_final['strategy'].replace(strategies_name)
shv = pd.DataFrame(port_returns_final.groupby('strategy')['return'].std()).T

#On elimine les date on ces deux coomne sint vide
merged_df = merged_df.dropna(subset=['R6M', 'target_6m'])

#On remplace les valeurs manquentes par leurs valeurs précédente
merged_df['Div_yield'] = merged_df['Div_yield'].fillna(method='ffill')
merged_df['Payout'] = merged_df['Payout'].fillna(method='ffill')
merged_df['PER'] = merged_df['PER'].fillna(method = 'ffill')

#Divise notre base de donnée on base d'entrainement et de test 
train_df = merged_df[merged_df['Date'] <= pd.Timestamp('2023-06-14')]
test_df = merged_df[merged_df['Date'] > pd.Timestamp('2023-06-14')]


#Selctionne les facteurs
features= merged_df.drop(columns=['Ticker', 'Date', 'Month','Close', 'Year','target_1m']).columns



#On applique une régression linéaire 
import statsmodels.api as sm
y_train=train_df['target_1m'].values
X_train=train_df[features].values
X_train =sm.add_constant(X_train) 
model_reg = sm.OLS(y_train, X_train).fit() 


print(model_reg.summary())

X_test = test_df[features].values
X_test = sm.add_constant(X_test)
y_test = test_df['target_1m'].values

mse=np.mean((model_reg.predict(X_test)-y_test)**2)
print(f'MSE: {mse}')

hitratio=np.mean(model_reg.predict(X_test)*y_test>0)
print(f'Hit Ratio: {hitratio}')

y_penalized_train=train_df['target_1m'].values 
X_penalized_train=train_df[features].values 
model = ElasticNet(alpha=0.1, l1_ratio=0.1)
fit_pen_pred=model.fit(X_penalized_train,y_penalized_train)

y_penalized_test = test_df['target_1m'].values 
X_penalized_test = test_df[features].values 
mse=np.mean((fit_pen_pred.predict(X_penalized_test)-y_penalized_test)**2)
print(f'MSE: {mse}')

hitratio=np.mean(fit_pen_pred.predict(X_penalized_test)*y_penalized_test>0)
print(f'Hit Ratio: {hitratio}')

from statsmodels.api import OLS , add_constant

selected_features = np.where(model.coef_ != 0)[0]
X_selected = X_penalized_train[:, selected_features]

# Ajustement d'un modèle OLS pour les variables sélectionnées
X_selected = add_constant(X_selected)  # Ajout de l'ordonnée à l'origine
ols_model = OLS(y_penalized_train, X_selected).fit()

# Résultats du modèle OLS
print(ols_model.summary())

print("ElasticNet Model Summary")
print("-" * 30)
print(f"Alpha (regularization parameter): {model.alpha}")
print(f"L1 ratio: {model.l1_ratio}")
print(f"Number of iterations: {model.n_iter_}")
print(f"Coefficients: {model.coef_}")
print(f'P-valeur de chaque coefficient : {model.p}')
print(f"Intercept: {model.intercept_}")
print(f"R^2 score (on training data): {model.score(X_penalized_train, y_penalized_train)}")
print(f"R^2 score (on testing data): {model.score(X_penalized_test, y_penalized_test)}")



from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error

np.random.seed(42)
tscv = TimeSeriesSplit(n_splits=5)  #Numéro de split
model = ElasticNet()
param_grid = {
    'alpha': [0.1, 0.2,0.5, 1.0, 5.0],
    'l1_ratio': [0.1, 0.2,0.5, 0.9]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',  #Le critère de 
    n_jobs=-1  #Utiliser tous les processeurs 
)
X = merged_df[features].values
y = merged_df['target_1m'].values

grid_search.fit(X, y)
print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleure Score (Negative MSE):", grid_search.best_score_)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X)
mse = mean_squared_error(y, predictions)
print("Mean Squared Error sur tous les données:", mse)



import datetime as dt
from datetime import datetime
sep_oos = '2022-12-31'
ticks = list(merged_df['Ticker'].unique())
N = len(ticks)
t_oos = merged_df['Date'].unique()
t_as = list(returns.index.values)
Tt = len(t_oos)
nb_port = 2
portf_weights = np.zeros(shape=(Tt, nb_port, 14242))
portf_returns = np.zeros(shape = (Tt, nb_port))

def weights_elastic(train_data, test_data, features):
    train_features = train_data[features]
    train_label = train_data['target_1m']

    model = ElasticNet(alpha = 0.1, l1_ratio = 0.1)
    model.fit(train_features, train_label)
    pred = model.predict(test_data[features])
    w_names = test_data['Ticker']
    w = pred > np.median(pred)
    w = w/np.sum(w)
    return w, w_names
    
        
    
    

def portf(train_data, test_data, features, j):
    if j == 0:
        N = len(test_data['Ticker'])
        w = np.repeat(1/N, N)
        w_names = test_data['Ticker']
        
        return w, w_names
    elif j ==1:
        return weights_elastic(train_data, test_data, features)
    

m_offset = 1
train_size = 1
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
t_oos = pd.to_datetime(t_oos)
all_stock_names = np.array(test_data['Ticker'])  # Example

for t in range(len(t_oos)-1):
    ind = (merged_df['Date'] < t_oos[t]-dt.timedelta(m_offset*30)) & (merged_df['Date'] > t_oos[t]-dt.timedelta(m_offset*30)-dt.timedelta(365*train_size))

    train_data= merged_df.loc[ind,:] # Trainsample
    if train_data.empty:
        print(f"No training data for t={t}. Skipping.")
        continue
    test_data= merged_df.loc[merged_df['Date'] == t_oos[t],:] # Testsample
    if test_data.empty:
        print(f"No test data for t={t}. Skipping.")
        continue
    realized_returns= test_data["target_1m"]
    if len(realized_returns) == 0:
        print(t_oos[t])
    all_stock_names = np.array(test_data['Ticker'])  # Example

    # Computingreturnsvia:1Mholdingperiod!
    # Example mapping stock names to indices

    for j in range(nb_port):
        temp_weights, stocks = portf(
        train_data, test_data,features,j)
    
        stocks_indices = np.where(np.isin(all_stock_names, stocks))[0]
    # Weights
        portf_weights[t,j,stocks_indices] = temp_weights
        portf_returns[t,j] = np.sum(temp_weights * realized_returns)# Allocateweights

def turnover(weights, asset_returns, t_oos,):
    turn = 0
    for t in range(1,len(t_oos)):
        realised_returns = asset_returns[returns.index == t_oos[t]].values
        prior_weights  = weights[t-1][:20]*(1+realised_returns)
        turn = turn +np.sum(np.abs(weights[t][:20]- prior_weights/np.sum(prior_weights)))
        return turn/(len(t_oos)-1)

def perf(port_returns, weights, asset_returns, t_oos):
    avg_ret = np.nanmean(port_returns)
    vol = np.nanstd(port_returns, ddof = 1)
    sharp = avg_ret/vol
    Var = np.quantile(port_returns, 0.05)
    turn = turnover(weights, asset_returns, t_oos)
    met = [avg_ret, vol, sharp, Var, turn]
    return met


def perf_met_multi(port_returns, weights, asset_returns, t_oos, strat_name):
    J = weights.shape[1]
    met = []
    for j in range(J):
        temp_met = perf(port_returns[:,j], weights[:,j,:], asset_returns, t_oos)
        met.append(temp_met)
    return pd.DataFrame(met,index = strat_name, columns = ['avg_ret', 'Vol', 'Sharp Ratio','VaR 5', 'turn'])

asset_returns = merged_df[['Date', 'Ticker', 'target_1m']].pivot(
index='Date', columns='Ticker',values='target_1m')

# Zeroreturnsformissingpoints
print(perf_met_multi(portf_returns,portf_weights,
asset_returns,t_oos,strat_name=["EW","Elastic"]))


