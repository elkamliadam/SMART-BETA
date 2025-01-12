
import pandas as pd 
import numpy as np 

fund = pd.read_excel('Fun_data.xlsx')
# Néttoyage des colomnes de type numérique de caractères spéciaux
fund[2021] = fund[2021].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2022] = fund[2022].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
fund[2023] = fund[2023].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)

#Enlève les valeurs manquantes dans la colonne Ticker
fund['Ticker'] = fund['Ticker'].fillna(method='ffill')
print('Extrait de la base de donnée fund---------------')
print(fund)

#Extraire le PBR pour le calcul des SMB
PBR = fund[fund['Ratio'] == 'PBR']
print('Extrait de la base PBR------------------------')
print(PBR)

#Inverse les PBR pour obtenir le B/M ratio
PBR[2023] = 1 / PBR[2023]
PBR[2022] = 1 / PBR[2022]
PBR[2021] = 1 / PBR[2021]


#Corriger quelque faute orthographique dans les tickers
PBR['Ticker'][5] = 'AFM'
PBR['Ticker'][149] = 'DIS'
PBR['Ticker'][263] = 'MNG'



#Catégorise les actions en 3 groupes: small, mid et big selon leur B/M ratio
#La décision est fait suivant le 30% quantile et le 70% quantile
def stock_size(year):
    quantile_30 = PBR[year].quantile(0.3)
    quantile_70 = PBR[year].quantile(0.7)
    small = PBR[PBR[year] <= quantile_30]['Ticker']
    big = PBR[PBR[year] >= quantile_70]['Ticker']
    mid = PBR[(PBR[year] > quantile_30) & (PBR[year] < quantile_70)]['Ticker']
    return small, big, mid


#Extraire la capitalisatin moyenne de chaque action sur chaque année pour chaque action
Tickers = fund['Ticker'].unique()
Tickers[0] = 'AFM'
Tickers[24] = 'DIS'
Tickers[43] = 'MNG'
#Création de la base Capi pour la capitalisation moyenne
Capi = pd.DataFrame()
for tick in Tickers :
    data = pd.read_excel(f'Stocks//{tick}.xlsx')
    data['Séance'] = pd.to_datetime(data['Séance'], format = '%d/%m/%Y')
    data['Year'] = data['Séance'].dt.year
    Capi[tick] = data.groupby('Year')['Capitalisation'].mean()



Capi = Capi.T 
print('Extrait de la base de Capitalisation --------------------')
print(Capi)

#Catégorisation des actions selon les deux caractéristqiues (B/M ratios et leurs capitalisation)
Portfolio = pd.DataFrame(columns= Tickers)
Portfolio['Year'] = [2021, 2022, 2023]
Portfolio.index = [2021, 2022, 2023]
Year = [2021, 2022, 2023]
for tick in Tickers:
    for year in Year:
        small, mid, big = stock_size(year)
        small = list(small)
        mid = list(mid)
        big = list(big)
        if tick in small :
            if Capi.loc[tick, year] < Capi[year].mean():
                Portfolio.loc [year, tick] = 'S-S'
            else : 
                Portfolio.loc[year, tick] = 'S-B'
        elif tick in mid:
            if Capi.loc[tick, year] < Capi[year].mean():
                Portfolio.loc[ year, tick] = 'M-S'
            else : 
                Portfolio.loc[ year, tick] = 'M-B'
        elif tick in big:
            if Capi.loc[tick, year] < Capi[year].mean():
                Portfolio.loc[ year, tick] = 'B-S'
            else : 
                Portfolio.loc[ year, tick] = 'B-B'
# S-S : Small B/M Small Cap
# S-B : Small B/M Big Cap
# M-S : Medium B/M Small Cap
# M-B : Medium B/M Big Cap
# B-S : Big B/M Small Cap
# B-B : Big B/M Big Cap

#Puisqu'on n'a pas le B/M de l'année 2024 on va utiliser celle de 2023 avec la capitalisation de 2024
for tick in Tickers:
        small, mid, big = stock_size(2023)
        small = list(small)
        mid = list(mid)
        big = list(big)
        if tick in small :
            if Capi.loc[tick, 2024] < Capi[2024].mean():
                Portfolio.loc [2024, tick] = 'S-S'
            else : 
                Portfolio.loc[2024, tick] = 'S-B'
        elif tick in mid:
            if Capi.loc[tick, 2024] < Capi[2024].mean():
                Portfolio.loc[ 2024, tick] = 'M-S'
            else : 
                Portfolio.loc[ 2024, tick] = 'M-B'
        elif tick in big:
            if Capi.loc[tick, 2024] < Capi[2024].mean():
                Portfolio.loc[ 2024, tick] = 'B-S'
            else : 
                Portfolio.loc[ 2024, tick] = 'B-B'


print("Extrait de la catégorie de chaque action sur chaque année d'après le B/M et la capitalization -----------")
print(Portfolio)

#Organiser pour chaque année les actions de même types
ss_tickers = {}
sb_tickers = {}
ms_tickers = {} 
mb_tickers = {}
bb_tickers = {}
bs_tickers = {}
annee = [2021, 2022, 2023, 2024]
for year in annee:
    ss_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'S-S')].tolist()
    sb_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'S-B')].tolist()
    ms_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'M-S')].tolist()
    mb_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'M-B')].tolist()
    bb_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'B-B')].tolist()
    bs_tickers[year] = Portfolio.columns[(Portfolio.loc[year] == 'B-S')].tolist()

print('Exemple de catégorie small B/M small Cap \n')
print(ss_tickers)


#Création de la base de donnée de tous les prix de cloture des 72 société coté au Maroc
Price = pd.DataFrame(columns= Tickers)
for tick in Tickers : 
    data = pd.read_excel(f'Stocks//{tick}.xlsx')
    Price[tick] = data['Dernier Cours']

#On index la base Price avec les dates de jours de trading qu'on récupère de AFMA et on ajoute une colonne Year pour afficher les année
Test = pd.read_excel('Stocks//AFM.xlsx')
Test['Séance'] = pd.to_datetime(Test['Séance'], format = '%d/%m/%Y')
Price.index = Test['Séance']
Price['Year'] = Price.index.year
    
print("-----------Extrait de la base de donnée des prix de clotures ------------- \n")
print(Price)
Price.to_csv('Price.csv')
#On calcul le rendement simple de chaque prix qu'on stock dans la base returns
#L'utilisatin des rendements logarithmique n'est pas nécessaire vu qu'on ne va pas utiliser de volatilité 
returns = Price.pct_change()

#On ajoute les colonne Year et Date pour mieux indexer notre base
returns['Year'] = Price['Year']
returns['Date'] = Price.index

print("------------------- Extrait de la base des rendements simple ---------------------\n")
print(returns)
returns.to_csv('Returns_fact.csv')
#On crée des nouvelle colonnes pour calculer le rendement quatidien de chaque type de portefeuille
for category in ['SS', 'SB', 'MS', 'MB', 'BB', 'BS']:
    returns[category] = np.nan

# On calcul le rendement simple de chaque portefeuille (equally weighted)
for year in annee:
    for category, tickers in zip(['SS', 'SB', 'MS', 'MB', 'BB', 'BS'], 
                                 [ss_tickers, sb_tickers, ms_tickers, mb_tickers, bb_tickers, bs_tickers]):
        if tickers[year]:
            returns[category][returns['Year'] == year] = returns[tickers[year]].mean(axis=1)


#On crée la base de donnée Factors ou on va stocker nos facteurs
Factors = pd.DataFrame()
#Facteur SMB comme différence entre les portefeuilles extreme de taille
Factors['SMB'] = -((returns['BS']+returns['BB'])/2) - ((returns['SS']+returns['SB'])/2)
#Facteur HML comme différence entre les portefeuille extremes de valeurs
Factors['HML'] = ((returns['SB']+returns['MB']+returns['BB'])/3) - ((returns['SS']+returns['MS']+returns['BS'])/3)



# On calcul les Rendement des 12 mois passés pour chaque action en excluant le moi le plus récent
for ticker in Tickers:
    returns[f'{ticker}_12m_excl_recent'] = returns[ticker].rolling(window=252, min_periods=1).sum().shift(21)


# Initialize le facteur WML qu'on va calucler comme la différence des deux portefeuillle extreme
#chaque portefeuuille est constitué de 21 action avec le plus grand rendement et les 21 action avec le plus petit rendement
Factors['WML'] = np.nan

# Calcul le WML pour chaque jour
for date in returns.index:
    # Pour une date deonné on prend les rendement des 12 mois passé
    past_returns = returns.loc[date, [f'{ticker}_12m_excl_recent' for ticker in Tickers]]
    
    # Elimine les valeurs NaN
    past_returns = pd.to_numeric(past_returns.dropna())
    
    # Vérifier qu'il y a au moins 42 tickers
    if len(past_returns) >= 42:
        # Get les 21 premiers actions
        top_21_tickers = past_returns.nlargest(21).index.str.replace('_12m_excl_recent', '')
        
        # Get les 21 dernier actions
        bottom_21_tickers = past_returns.nsmallest(21).index.str.replace('_12m_excl_recent', '')
        
        # Le rendement moyen des deux portefeuilles
        mean_top_21 = returns.loc[date, top_21_tickers].mean()
        mean_bottom_21 = returns.loc[date, bottom_21_tickers].mean()
        
        # Calcule le WML
        Factors.loc[date, 'WML'] = mean_top_21 - mean_bottom_21

#On enregistre les Return On Equity de chaque action pour les trois dernière années
ROE  = fund[fund['Ratio'] == 'ROE (en %)']
print("-------------------Extrait de la base des ROE-------------------\n")
print(ROE)

#On catégorise les actions entre ceux avec une grande profitabilité et ceu avec une petite portefeuille
#Les seuils sont prise comme le quantile 30% et e quantile 70%
def stock_profitability(year):
    quantile_30 = ROE[year].quantile(0.3)
    quantile_70 = ROE[year].quantile(0.7)
    small = ROE[ROE[year] <= quantile_30]['Ticker']
    big = ROE[ROE[year] >= quantile_70]['Ticker']
    return small, big



#On crée une base pour catégoriser a quel portefeuille apartient chaque action sur chaque année
Prof = pd.DataFrame(columns= Tickers)
Prof['Year'] = [2021, 2022, 2023]
Prof.index = [2021, 2022, 2023]
Year = [2021, 2022, 2023]
for tick in Tickers:
    for year in Year:
        small, big = stock_profitability(year)
        small = list(small)
        big = list(big)
        if tick in small :
                Prof.loc [year, tick] = 'Low'
        elif tick in big:
                Prof.loc[ year, tick] = 'High'
for tick in Tickers:
        small, big = stock_profitability(2023)
        small = list(small)
        big = list(big)
        if tick in small : 
                Prof.loc[2024, tick] = 'Low'
        elif tick in big : 
                Prof.loc[2024, tick] = 'High'

print("-----------------Extrait de la catgéorie de chaque action d'aptès sa profitabilité pour chaque année---------------\n")
print(Prof)

#On catégorise les action en deux
low_tickers = {}
high_tickers = {}
annee = [2021, 2022, 2023, 2024]
for year in annee:
    high_tickers[year] = Prof.columns[(Prof.loc[year] == 'High')].tolist()
    low_tickers[year] = Prof.columns[(Prof.loc[year] == 'Low')].tolist()

print("---------------Exemple des actions de faible profitabilité----------------\n")
print(low_tickers)



#On divise les actions en deux
for category in ['Low', 'High']:
    returns[category] = np.nan

# Calcule les valeurs des portefeuille extrêmes
for year in annee:
    for category, tickers in zip(['Low', 'High'], 
                                 [low_tickers, high_tickers]):
        if tickers[year]:
            returns[category][returns['Year'] == year] = returns[tickers[year]].mean(axis=1)



#Le facteur RMW et la différence des deux portefeuilles extremes
Factors['RMW'] = returns['High'] - returns['Low']

#On importe la séries des taux sans risque (Taux des bons de trésore à maturité de 10 ans)

rf = pd.read_csv('RF.csv')
rf['Dernier'] = rf['Dernier'].str.replace(' ', '').str.replace(',', '.').replace('-', np.nan).astype(float)
rf['Dernier'] = rf['Dernier']/100
#On importe l'indice MASI pour calculer le rendement ecédentaire par rapport aux taux sans risque 
#Le taux sans risque est pris comme le taux de bons de trésore pour chaque année à maturité 10 ans
MASI = pd.read_excel('MASI.xlsx')
MASI['séance'] = pd.to_datetime(MASI['séance'], format='%d/%m/%Y')


MASI['return'] = MASI['valeur indice'].pct_change()
MASI['factor'] = MASI['return'] - rf['Dernier']


#On ajoute le dernier facteur
Factors['MKT-RF'] = np.nan
for i in range(len(MASI)):
    Factors['MKT-RF'][i] = MASI['factor'][i]
print("---------------Extrait des facteurs------------------\n")
print(Factors)

#On enregistre notre base de donnée sous format CSV
Factors.to_csv('Factors.csv')



import matplotlib.pyplot as plt 

plt.figure(figsize=(20, 8))
plt.plot(Factors['SMB'], color = 'Red', label = 'SMB')
plt.plot(Factors['HML'], color = 'blue', label = 'HMB')
plt.plot(Factors['RMW'], color = 'green', label = 'RMW')
plt.plot(Factors['WML'], color = 'orange', label = 'WML')
plt.plot(Factors['MKT-RF'], color = 'grey', label = 'MKT-RF4')
plt.legend()
plt.xlabel('Date')
plt.title('Evolution quotidienne des facteurs Marocains', fontweight = 'bold')
plt.savefig('Facteurs.png')  
plt.show()






