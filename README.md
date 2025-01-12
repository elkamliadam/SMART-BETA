l'organisation du code est la suivante : 
\begin{enumerate}
    \item \textbf{Factors.py} : ce fichier calcul les facteurs et il donnes comme résultats : 
    \begin{itemize}
        \item la base de donnée \textit{Price.csv} : contient les prix quotidiens des 74 actions utilisées.
        \item la base de donnée \textit{Returns\_fact.csv} : contient les rendements quotidiens des 74 actions utilisées.
        \item la base de donnée \textit{Factors.csv} : contient l'évolution quotidienne des facteurs calculés.
        \item Figure \textit{Facteurs.png} : Visualization de l'évolution quotidienne des facteurs.
    \end{itemize}
    \item \textbf{Anomalies\_facteurs.py} : ce fichier permet la sélection des facteurs et il donne : 
    \begin{itemize}
        \item \textit{regression\_summaries\_daily.txt}  et \textit{regression\_summaries\_month.txt} : des fichiers text contenant le résultats de la régression de chaque action sur les facteurs (quotidien et mensuel) .
        \item \textit{Crossection\_regression.txt} \textit{Crossection\_regression\_month.txt} : Contient les résultats de la corssectional regression de la régression Fama Mcbeth(quotiden et mensuel).
        \item Figure \textit{daily\_factor\_premia.png} : figure de l'évolution quotidenne des risk premia.
        \item Afficher les p-values de chaque risk premia
        \item Base de donnée \textit{Ret\_mon.csv} : qui contient le rendement mensuel de chaque actions.
        \item Base de donnée \textit{Fact\_mon.csv} : qui contient les facteurs mensuels.
        \item \textit{Factor\_competition.txt} et \textit{Factor\_competition\_monthly.txt} : contient la régression de chaque facteur sur les autres pour l'etude de la compétition entre ces facteurs (au niveau quotidien et mensuel).
    
    \end{itemize}
    \item \textbf{Factor\_investing.py} : ce fichier contient le rest du calcul et la back testing et a part l'affichage des résultats il donne : 
    \begin{itemize}
        \item Figures \textit{Boxplot.png} le box plot de quelque facteurs importants, \textit{Corr\_matirx.png} la matrice de corrélation
    \end{itemize}
\end{enumerate}
