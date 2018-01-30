# Statistiques bayésiennes: Optimisation Bayésienne #

## Mode d'emploi ##

Pour lancer les algorithmes d'optimisation bayésienne, il faut se placer dans main:\
  0/ Exécuter les lignes 1-16 pour importer les packages\
  1/ Exécuter les lignes 17-45 pour fixer les paramètres.\
  2/ Exécuter les lignes 46-71 pour estimer p et theta par MLE\
  3/ Exécuter les lignes 72-110 pour plotter les 2 critères d'acquisition EI et LBC ainsi que la fonction mystère évaluée aux points initiaux\
  4/ Exécuter les lignes 111-118 pour lancer les itérations d'optimisation bayésienne.
      Pour changer la fonction d'acquisition il suffit de changer la ligne 115:\
      - acq_func1 pour utiliser l'EI\
      - acq_func2 pour utiliser la LBC\
  5/ Exécuter les lignes 119-125 pour récupérer les métriques d'évaluation de l'algorithme\
  
  Pour construire les tableaux de résultats du rapport avec plusieurs lancements de l'algorithme, il faut:\
    - exécuter le fichier "results" qui donne les métriques comparatives entre EI et LCB\
    - exécuter le fichier "results_etas" qui donne les métriques comparatives entre différents paramétrages de LCB\
    ATTENTION: ces algorithmes prennent un certain temps à tourner (environ 1min30 pour une optimisation avec 100 itérations)
    
NB: il se peut qu'une erreur de type "LinAlgError: Matrix is not positive definite" survienne: erreur lors de l'inversion de la matrice R. Il faut faire tourner de nouveau l'algorithme.
