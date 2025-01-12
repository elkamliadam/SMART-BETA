## Organisation du code

1. **Factors.py**  
   Ce fichier calcule les facteurs et génère les résultats suivants :  
   - Base de données `Price.csv` : contient les prix quotidiens des 74 actions utilisées.  
   - Base de données `Returns_fact.csv` : contient les rendements quotidiens des 74 actions utilisées.  
   - Base de données `Factors.csv` : contient l'évolution quotidienne des facteurs calculés.  
   - Figure `Facteurs.png` : visualisation de l'évolution quotidienne des facteurs.  

2. **Anomalies_facteurs.py**  
   Ce fichier permet la sélection des facteurs et génère :  
   - `regression_summaries_daily.txt` et `regression_summaries_month.txt` : fichiers texte contenant les résultats des régressions de chaque action sur les facteurs (quotidien et mensuel).  
   - `Crossection_regression.txt` et `Crossection_regression_month.txt` : résultats de la régression croisée Fama-MacBeth (quotidien et mensuel).  
   - Figure `daily_factor_premia.png` : évolution quotidienne des risk premia.  
   - Affichage des p-values pour chaque risk premia.  
   - Base de données `Ret_mon.csv` : rendements mensuels de chaque action.  
   - Base de données `Fact_mon.csv` : facteurs mensuels.  
   - `Factor_competition.txt` et `Factor_competition_monthly.txt` : régressions de chaque facteur sur les autres pour étudier la compétition entre facteurs (au niveau quotidien et mensuel).  

3. **Factor_investing.py**  
   Ce fichier contient les calculs restants et les backtests. Outre l'affichage des résultats, il génère :  
   - Figures `Boxplot.png` : boxplot de quelques facteurs importants, et `Corr_matrix.png` : matrice de corrélation.  
