# Plan du projet : Prévision de la demande de vélos en libre-service (Bike Sharing)

**Données :** `hour.csv` uniquement (~17 379 lignes, une ligne par heure, 2011-2012)
**Cible :** `cnt` (régression). Le dataset est une série temporelle, donc la validation doit respecter l'ordre chronologique.

---

## Phase 1 : Cadrage

- Travailler exclusivement sur **`hour.csv`** (granularité horaire, variable `hr` disponible).
- Définir la métrique d'évaluation : **RMSE**, **MAE**, **R²**, et idéalement **RMSLE** (la métrique classique de ce problème).
- Définir la stratégie de validation : découpage temporel (ex. entraînement sur 2011 + début 2012, test sur les derniers mois de 2012).
- Préciser l'objectif de prévision : avec ou sans connaissance de la météo future.

## Phase 2 : Exploration des données (EDA)

- Vérifier les types, valeurs manquantes, doublons, cohérence des dates.
- Vérifier la **continuité temporelle** : des heures sont absentes (nuits sans location), donc la série horaire n'est pas strictement régulière.
- Analyser la distribution de `cnt` (forte asymétrie, outliers, transformation `log1p`).
- Étudier la demande selon : `hr`, `weekday`, `mnth`, `season`, `yr`, `holiday`.
- Comparer `workingday` vs week-end : profils horaires très différents (pics à 8h et 17-18h les jours ouvrés, pic en milieu de journée le week-end).
- Analyser l'impact de la météo (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`). Noter que `weathersit = 4` est très rare.
- Matrice de corrélation et détection de **multicolinéarité** (`temp` et `atemp` quasi identiques).
- Comparer `casual` et `registered` (comportements distincts) pour la compréhension uniquement.
- Mesurer la croissance entre 2011 et 2012 (tendance).

## Phase 3 : Nettoyage et préparation

- Convertir `dteday` en datetime et construire un timestamp complet (`dteday` + `hr`).
- Supprimer les colonnes inutiles : `instant`, et **`casual` et `registered`** (fuite de données : `cnt = casual + registered`, à exclure des features).
- Traiter les outliers éventuels et les valeurs suspectes de `hum` (0).
- Dé-normaliser si besoin d'interprétation (`temp` × 41, `atemp` × 50, `hum` × 100, `windspeed` × 67).
- Découpage train/test chronologique.

## Phase 4 : Feature engineering

- **Variables cycliques** : encodage sin/cos pour `hr`, `mnth`, `weekday`.
- **Variables catégorielles** : one-hot encoding (`season`, `weathersit`) pour les modèles linéaires.
- **Variables d'interaction** : `workingday × hr`, `temp × hum`.
- **Variables temporelles** : heures de pointe, jour de l'année, week-end, tendance (`yr` ou index temporel).
- **Variables de retard (lags)** : `cnt` à t-1, t-24, t-168 (seulement pertinent en prévision à court terme ; à construire sur la série horaire reconstruite).
- **Moyennes glissantes** de la demande ou de la météo.

## Phase 5 : Modélisation

1. **Baselines** : moyenne par (`hr`, `weekday`), régression linéaire.
2. **Modèles linéaires régularisés** : Ridge, Lasso.
3. **Modèles à base d'arbres** : Random Forest, Gradient Boosting.
4. **Boosting performant** : XGBoost, LightGBM, CatBoost.
5. **(Optionnel)** Séries temporelles : SARIMA, Prophet, ou LSTM.
6. Tester la transformation `log1p(cnt)` pour stabiliser la variance.

## Phase 6 : Validation et optimisation

- Validation croisée temporelle (`TimeSeriesSplit`).
- Réglage des hyperparamètres (GridSearch, RandomSearch ou Optuna).
- Tableau récapitulatif de comparaison des modèles.
- Éventuellement un ensemble (stacking/blending) des meilleurs modèles.

## Phase 7 : Évaluation et interprétation

- Métriques finales sur le jeu de test.
- Analyse des résidus (par heure, saison, météo, jour ouvré vs week-end).
- **Importance des variables** (feature importance, SHAP).
- Courbes prédiction vs réel (journée type, semaine type).

## Phase 8 : Livrables

- Notebook structuré et commenté.
- Rapport/présentation avec conclusions métier (dimensionnement de la flotte, redistribution des vélos aux heures de pointe).
- (Optionnel) Sauvegarde du modèle (`joblib`) et mini-application de démonstration (Streamlit).

---

## Calendrier indicatif

| Étape | Durée estimée |
|---|---|
| Cadrage + EDA | 2-3 jours |
| Préparation + features | 2 jours |
| Modélisation + baselines | 3 jours |
| Optimisation + validation | 2-3 jours |
| Interprétation + rapport | 2 jours |

## Points d'attention

- **Fuite de données** : exclure `casual` et `registered`.
- **Ordre temporel** : ne jamais mélanger aléatoirement train/test.
- **Heures manquantes** : à gérer avant de calculer lags et moyennes glissantes.
- **Tendance 2011 → 2012** : le modèle doit l'absorber (variable `yr`) ou être évalué sur une période récente.
- **Objectif de prévision** : sans prévisions météo futures, les features météo ne sont pas utilisables en vrai déploiement.