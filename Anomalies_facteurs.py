
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import statistics
from scipy.stats import t
#Import la base des facteurs et on visualise les facteurs aggrégé annuellement
factors = pd.read_csv('Factors.csv')
factors['Séance'] = pd.to_datetime(factors['Séance'], format = '%Y-%m-%d')
factors['Years'] = factors['Séance'].dt.year
fac = factors.drop(columns=['Séance'])
fac.groupby('Years').mean().plot()


#On import la base de donnée des facteurs fondamentaux des entreprises
fund = pd.read_excel('Fun_data.xlsx')
fund[2021] = fund[2021].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2022] = fund[2022].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2023] = fund[2023].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)

#Enlève les valeurs manquantes dans la colonne Ticker
fund['Ticker'][0] = 'AFM'
fund['Ticker'][144] = 'DIS'
fund['Ticker'][258] = 'MNG'
fund['Ticker'] = fund['Ticker'].fillna(method = 'ffill')

#Stock le ticker de chaque action 
Ticker = fund['Ticker'].unique()


#On importe la base de donnée des prix de clotûre de toutes les actions de la bourse de CasaBlanca
Close = pd.read_csv('Price.csv')


#Fama Mcbeth Regression sur les rendements quotidien
Factors = pd.read_csv('Factors.csv')

#On calcul les rendements simples 
#On n'a pas intérêt à utiliser les rendement logarithmique puisque on ne travaillera pas avec la volatilité
Returns = Close.drop(columns=['Séance', 'Year'])

Returns = Returns.pct_change()

#Etape 01 : Régression de chaque action sur les facteurs et stockage des Betas obtenus
betas = []
Tickers = Returns.columns
X = Factors[['SMB', 'HML', 'WML', 'RMW', 'MKT-RF']]
#On enregistre les détails de chaque régression dans le fichier Text 'regression_summaries_daily.txt'
with open('regression_summaries_daily.txt', 'w') as f:    
    for tick in Tickers:
        y = Returns[tick]
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop').fit()
        f.write(f'Summary for {tick} : \n')
        f.write(model.summary().as_text())
        f.write('\n\n')
        betas.append(model.params)    

#On stocke les Betas de chaque facteurs pour passer à l'étape 02
SMB = []
HML = []
WML = []
RMW = []
MKT = []
for i in range(len(betas)):
    SMB.append(betas[i][1])
    HML.append(betas[i][2])
    WML.append(betas[i][3])
    RMW.append(betas[i][4])
    MKT.append(betas[i][5])

#Etape 02 : Cross asset regression 
#On va regresser pour chaque instant t les rendements de tous les actions sur les betas obtenus de chaque facteurs
#Cross sectional Regression
data = {
    'SMB': SMB,
    'HML': HML,
    'WML': WML,
    'RMW': RMW,
    'MKT-RF': MKT
}

data = pd.DataFrame(data)

#On stocke les Gamma (Risk premia de chaque facteurs) de la cross sectional regression
Gamma = []
X = data[['SMB', 'HML', 'WML', 'RMW', 'MKT-RF']]
#On enregistre les détails de chaque régression dans le fichier text 'Crosssectional-regression.txt'
with open('Crosssectional_regression.txt', 'w') as f:
    for i in range(len(Returns)):
        X = sm.add_constant(X)
        y = Returns.iloc[i].values
        model = sm.OLS(y,X).fit()
        f.write(model.summary().as_text())
        f.write('\n\n')
        Gamma.append(model.params)


#Etape 03 : On calcule la t-statistique et leurs p-valeurs de chaque facteurs

SMB_premia = []
HML_premia = []
WML_premia = []
RMW_premia = []
MKT_premia = []
for i in range(len(Gamma)):
    SMB_premia.append(Gamma[i][1])
    HML_premia.append(Gamma[i][2])
    WML_premia.append(Gamma[i][3])
    RMW_premia.append(Gamma[i][4])
    MKT_premia.append(Gamma[i][5])



date = Close['Séance']
date = pd.to_datetime(date, format = '%Y-%m-%d')
len(date)
#On visualise l'évolution quotidienne de ces risk premia
plt.figure(figsize=(20, 10))
plt.plot(date, SMB_premia, label='SMB Premia')
plt.plot(date, HML_premia, label='HML Premia')
plt.plot(date, MKT_premia, label='MKT Premia')
plt.plot(date, RMW_premia, label='RMW Premia')
plt.plot(date, WML_premia, label='WML Premia')
plt.title('Evolution Quotidienne des Primes de risque de chaque facteurs', fontweight = 'bold')
plt.xlabel('Date')
plt.ylabel('Factor Premia')
plt.legend()
plt.grid(True)
plt.savefig('daily_factor_premia.png')
plt.show()


#On construit les statistique
#Calcul des moyen
SMB_stat = np.nanmean(SMB_premia,)
HML_stat = np.nanmean(HML_premia)
RMW_stat = np.nanmean(RMW_premia)
WML_stat = np.nanmean(WML_premia)
MKT_stat = np.nanmean(MKT_premia)

#l'écart type
SMB_sd = np.nanstd(SMB_premia, ddof = 1)
HML_sd = np.nanstd(HML_premia, ddof = 1)
RMW_sd = np.nanstd(RMW_premia, ddof = 1)
WML_sd = np.nanstd(WML_premia, ddof = 1)
MKT_sd = np.nanstd(MKT_premia, ddof = 1)
#T-statistique
SMB_tstat = SMB_stat/(SMB_sd/np.sqrt(len(Gamma)))
HML_tstat = HML_stat/(HML_sd/np.sqrt(len(Gamma)))
RMW_tstat = RMW_stat/(RMW_sd/np.sqrt(len(Gamma)))
WML_tstat = WML_stat/(WML_sd/np.sqrt(len(Gamma)))
MKT_tstat = MKT_stat/(MKT_sd/np.sqrt(len(Gamma)))

#Calcul des p-values
df = len(Gamma)-1
p_smb = 2*t.sf(abs(SMB_tstat), df)
p_hml = 2*t.sf(abs(HML_tstat), df)
p_rmw = 2*t.sf(abs(RMW_tstat), df)
p_wml = 2*t.sf(abs(WML_tstat), df)
p_mkt = 2*t.sf(abs(MKT_tstat), df)
#On affiche les p-values
print('SMB : ', p_smb, '\n')
print('HML :',p_hml, '\n')
print('MKT-RF : ',p_mkt, '\n')
print('RMW : ',p_rmw, '\n')
print('WML : ',p_wml, '\n')


#Compétition des facteurs quotidien
# Drop la colonne 'Séance' 
drop = ['Séance', 'Years']
factors_data = factors.drop(columns=drop)
#On enregistre le résultat de la régression de chaque facteur sur les autres 
#dans un fichier text nomé 'Factor_competition.txt'
with open('Factor_competition.txt', 'w') as f:
    for column in factors_data.columns:
    #Définire la variable à expliquer
        y = factors_data[column]
    
    #Définire les variables explicatives
        X = factors_data.drop(columns=[column])
        X = sm.add_constant(X)
    
    # Entraîne le modèle de régression
        model = sm.OLS(y, X, missing='drop').fit()
    
    # Enregistre les données dans notre fichier text
        f.write(f'{column} regression \n')
        f.write(model.summary().as_text())
        f.write('\n\n')
        

#Aggrégation Mensuel
#Dans cette partie on va refaire la régression Frensh Mcbeth et la compétition des facteurs 
# Mais cette fois ci pour les donnée mensuelle
Factors['Séance'] = pd.to_datetime(Factors['Séance'], format = '%Y-%m-%d')
Factors['month'] = Factors['Séance'].dt.to_period('M')
numeric_columns = Factors.select_dtypes(include=[np.number]).columns
#On calcul les rendement mensuelle à partir des rendements quotidien
for col in numeric_columns:
    if Factors[col].isna().all():
        print(f"Column '{col}' contains only NaN values and will be skipped.")
        continue
    Factors[f'{col}_month'] = Factors.groupby('month')[col].transform(
        lambda x: np.prod(1 + x.dropna()) - 1 if not x.isna().all() else np.nan
    )
    


Factor_month = Factors[['month'	,'SMB_month'	,'HML_month'	,'WML_month',	'RMW_month',	'MKT-RF_month' ]]
Factor_month_data = Factor_month.drop(columns=['month'])
# On enregistre la régression de chaque facteur sur les autres dans un fichier text 'Factor_competition_monthly.txt'
with open('Factor_competition_monthly.txt', 'w') as f:
    for column in Factor_month_data.columns:
        y = Factor_month_data[column]
        X = Factor_month_data.drop(columns=[column])
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop').fit()
        f.write(f'{column} regression \n')
        f.write(model.summary().as_text())
        f.write('\n\n')

#On calcul les rendements mensuelles de nos actions
Close['Séance'] = pd.to_datetime( Close['Séance'], format = '%Y-%m-%d')
Close['month'] = Close['Séance'].dt.to_period('M')
Returns['month'] = Close['month']





numeric_columns_ret = Returns.select_dtypes(include=[np.number]).columns

for col in numeric_columns_ret:
    if Returns[col].isna().all():
        print(f"Column '{col}' contains only NaN values and will be skipped.")
        continue
    Returns[f'{col}_month'] = Returns.groupby('month')[col].transform(
        lambda x: np.prod(1 + x.dropna()) - 1 if not x.isna().all() else np.nan
    )
    




# On stock les rendements mensuelle dans la bse returns_monthly
returns_monthly = Returns.filter(like='_month')


#On refait la première étapes de la régression Fama Mcbeth
betas_month = []
Tickers = returns_monthly.columns
X = Factor_month[['SMB_month', 'HML_month', 'WML_month', 'RMW_month', 'MKT-RF_month']]
#On enregistre les résultats de chaque régression dans un fichier texte 'regression_summaries_month.txt'
with open('regression_summaries_month.txt', 'w') as f:    
    for tick in Tickers:
        y = returns_monthly[tick]
        X = sm.add_constant(X)
        model = sm.OLS(y, X, missing='drop').fit()
        f.write(f'Summary for {tick} : \n')
        f.write(model.summary().as_text())
        f.write('\n\n')
        betas_month.append(model.params)
        

SMB_month = []
HML_month = []
WML_month = []
RMW_month = []
MKT_month = []
for i in range(len(betas_month)):
    SMB_month.append(betas_month[i][1])
    HML_month.append(betas_month[i][2])
    WML_month.append(betas_month[i][3])
    RMW_month.append(betas_month[i][4])
    MKT_month.append(betas_month[i][5])


data_month = {
    'SMB': SMB_month,
    'HML': HML_month,
    'WML': WML_month,
    'RMW': RMW_month,
    'MKT-RF': MKT_month
}

data_month = pd.DataFrame(data_month)


#On exécute l'étape 02 de la régression Fama Mcbeth
Gamma_month = []
X = data_month[['SMB', 'HML', 'WML', 'RMW', 'MKT-RF']]
#On enregistre les détail de chaque cross sectional reression dans un fichier text 
# 'Crosssection_regression_month.txt'
with open('Crosssectional_regression_month.txt', 'w') as f:
    for i in range(len(returns_monthly)):
        X = sm.add_constant(X)
        y = returns_monthly.iloc[i].values
        model = sm.OLS(y,X, missing = 'drop').fit()
        f.write(model.summary().as_text())
        f.write('\n\n')
        Gamma_month.append(model.params)



#On commence le calcul des t-statistics
SMB_premia_month = []
HML_premia_month = []
WML_premia_month = []
RMW_premia_month = []
MKT_premia_month = []
for i in range(len(Gamma_month)):
    SMB_premia_month.append(Gamma_month[i][1])
    HML_premia_month.append(Gamma_month[i][2])
    WML_premia_month.append(Gamma_month[i][3])
    RMW_premia_month.append(Gamma_month[i][4])
    MKT_premia_month.append(Gamma_month[i][5])




SMB_stat_month = np.nanmean(SMB_premia_month)
HML_stat_month = np.nanmean(HML_premia_month)
RMW_stat_month = np.nanmean(RMW_premia_month)
WML_stat_month = np.nanmean(WML_premia_month)
MKT_stat_month = np.nanmean(MKT_premia_month)


SMB_sd_month = np.nanstd(SMB_premia_month, ddof = 1)
HML_sd_month = np.nanstd(HML_premia_month, ddof = 1)
RMW_sd_month = np.nanstd(RMW_premia_month, ddof = 1)
WML_sd_month = np.nanstd(WML_premia_month, ddof = 1)
MKT_sd_month = np.nanstd(MKT_premia_month, ddof = 1)

SMB_tstat_month = SMB_stat_month/(SMB_sd_month/np.sqrt(len(Gamma_month)))
HML_tstat_month = HML_stat_month/(HML_sd_month/np.sqrt(len(Gamma_month)))
RMW_tstat_month = RMW_stat_month/(RMW_sd_month/np.sqrt(len(Gamma_month)))
WML_tstat_month = WML_stat_month/(WML_sd_month/np.sqrt(len(Gamma_month)))
MKT_tstat_month = MKT_stat_month/(MKT_sd_month/np.sqrt(len(Gamma_month)))


df = len(Gamma_month)-1
print(df)
p_smb_month = 2*t.sf(abs(SMB_tstat_month), df)
p_hml_month = 2*t.sf(abs(HML_tstat_month), df)
p_rmw_month = 2*t.sf(abs(RMW_tstat_month), df)
p_wml_month = 2*t.sf(abs(WML_tstat_month), df)
p_mkt_month = 2*t.sf(abs(MKT_tstat_month), df)
#On affiche les p-value de chaque facteurs mensuel
print('SMB_mensuel : ',  p_smb_month, '\n')
print('HML_mensuel : ',p_hml_month, '\n')
print('MKT-RF_mensuel : ',p_mkt_month, '\n')
print('RMW_mensuel : ',p_rmw_month, '\n')
print('WML_mensuel : ',p_wml_month)


#On exporte les bases de donnée mensuelles des facteurs et des rendemenst 
returns_monthly.to_csv('Ret_mon.csv')
Factor_month.to_csv('Fact_mon.csv')