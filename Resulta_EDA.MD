# Résultats et décisions de l'EDA — Bike Sharing (`hour.csv`)

**Cible :** `cnt` (locations de vélos par heure) — 17 379 lignes × 17 colonnes.

> Les valeurs lues sur les graphiques sont **approximatives**. Les points de l'EDA dont les résultats n'ont pas encore été fournis sont listés en section 6.

---

## 1. Matrice de corrélation (Pearson)

### Résultats

| Variable | Corrélation avec `cnt` | Lecture |
|---|---|---|
| `temp` | 0,40 | Plus il fait chaud, plus la demande augmente |
| `atemp` | 0,40 | Identique à `temp` |
| `hr` | 0,39 | Lien partiel : la relation réelle est non linéaire (deux pics) |
| `hum` | −0,32 | Plus l'humidité est élevée, plus la demande baisse |
| `yr` | 0,25 | Effet de croissance 2011 → 2012 |
| `season` | 0,18 | Effet saisonnier modéré |
| `weathersit` | −0,14 | Mauvais temps, moins de locations |
| `mnth` | 0,12 | Faible |
| `windspeed` | 0,09 | Faible |
| `weekday`, `workingday`, `holiday` | ≈ 0,03 | Pas de lien linéaire direct (l'effet passe par la forme de la courbe horaire, voir section 3) |

### Multicolinéarité

| Paire | Corrélation | Constat |
|---|---|---|
| `temp` / `atemp` | 0,99 | Quasi identiques |
| `mnth` / `season` | 0,83 | Redondantes, peu gênant pour les arbres |
| `hum` / `weathersit` | 0,42 | Logique : plus d'humidité, plus de pluie |
| `yr` / variables météo | ≈ 0 | `yr` capte une croissance pure, indépendante de la météo |

---

## 2. Distribution de `cnt`

### Résultats

- Forte **asymétrie à droite** : beaucoup de petites valeurs (près de 3 000 heures dans la première classe) et une longue queue jusqu'à environ 1 000.
- **Médiane** autour de 140 locations par heure.
- Le boxplot signale des outliers au-delà d'environ 640. Ce ne sont **pas des erreurs** : ce sont les heures de pointe (section 3).
- `log1p(cnt)` resserre la queue mais reste irrégulier : un plateau entre 2 et 3,5 (régime de nuit, faible demande) et un pic vers 5,4 (régime de jour).

---

## 3. Profil horaire selon le type de jour

### Résultats

| Type de jour | Forme du profil | Valeurs approximatives |
|---|---|---|
| **Jour ouvré** | Deux pics nets : matin et soir (trajets domicile-travail) | ≈ 475 à 8h, ≈ 525 à 17h, ≈ 490 à 18h |
| **Week-end** | Grande bosse en milieu de journée | ≈ 380 entre 12h et 15h |
| **Férié** | Allure proche du week-end, niveau plus bas, petit relief à 8h | ≈ 300 au maximum |

- La nuit (0h à 3h), le week-end est plus actif que les jours ouvrés.
- L'effet de l'heure **dépend du type de jour** : c'est le graphique le plus important de l'EDA.

---

## 4. Météo et tendance (heures de pointe, jours ouvrés)

### Résultats

| `weathersit` | Médiane de `cnt` aux heures de pointe |
|---|---|
| 1 (clair) | ≈ 540 |
| 2 (brume / nuageux) | ≈ 455 |
| 3 (pluie / neige légère) | ≈ 235 |
| 4 (conditions extrêmes) | 1 seul point observé |

- Une pluie légère **divise la demande par plus de deux**.
- **Température :** la demande monte avec `temp` jusqu'à environ 0,7-0,8 (soit 29 à 33 °C), puis plafonne. Effet **non linéaire**.
- **Année :** à température égale, 2012 est nettement au-dessus de 2011.
- Forte dispersion à température fixe : l'heure et la météo expliquent une grande part du reste.

---

## 5. Décisions pour les phases suivantes

| # | Décision | Raison | Phase |
|---|---|---|---|
| 1 | **Garder tous les outliers** de `cnt` | Ce sont les heures de pointe, justement à prédire | 3 |
| 2 | **Tester `log1p(cnt)`** comme cible | Réduit l'asymétrie ; à comparer avec la cible brute (surtout modèles linéaires et RMSLE) | 5 |
| 3 | **Supprimer `atemp`**, garder `temp` | Corrélation de 0,99 | 3 |
| 4 | **Supprimer `casual`, `registered`,`dteday` , `instant`** | Fuite de données (`cnt = casual + registered`) et identifiant | 3 |
| 5 | **Créer `workingday × hr`** (ou une variable `day_type` : ouvré / week-end / férié) | Les profils horaires diffèrent fortement | 4 |
| 6 | **Encoder `hr` de façon cyclique** (sin/cos), éventuellement en plus de la valeur brute pour les arbres | Relation non linéaire, 23h proche de 0h | 4 |
| 7 | **Fusionner `weathersit = 4` avec `3`** | Un seul point : trop rare pour être appris | 3 |
| 8 | **Conserver `hum` et `weathersit`** malgré leur corrélation de 0,42 | Corrélation modérée, apport propre de chacune | 4 |
| 9 | **Utiliser des modèles à base d'arbres / boosting** en priorité | Relations non linéaires (heure, température) et interactions | 5 |
| 10 | **Gérer explicitement la tendance 2011 → 2012** | `yr` pèse 0,25 sur `cnt` ; les arbres ne savent pas extrapoler, donc le découpage train/test doit en tenir compte | 3 / 6 |

---

## 6. Points de l'EDA restant à analyser

Résultats non encore fournis :

| Point | Ce qu'on veut savoir | Décision attendue |
|---|---|---|
| Types, valeurs manquantes, doublons, cohérence des dates | Nombre de lignes incohérentes par test | Corrections éventuelles (Phase 3) |
| Continuité temporelle | Nombre d'heures et de jours absents, plus longs trous | Reconstruction de la grille horaire avant les lags |
| `windspeed = 0` et `hum = 0` | Nombre de valeurs suspectes | Imputation ou conservation |
| Demande selon `hr`, `weekday`, `mnth`, `season`, `yr`, `holiday` | Graphiques en barres | Vérifier notamment le sens de `season` (1 = hiver ?) |
| `casual` vs `registered` | Profils et sensibilité à la météo | Compréhension uniquement |
| Croissance 2011 → 2012 | Pourcentage de croissance par mois, évolution du profil horaire | Choix de la stratégie de validation et de la variable de tendance |

---

## 7. Résumé en une phrase

La demande est surtout structurée par **l'heure et le type de jour** (deux pics les jours ouvrés, une bosse le week-end), modulée par **la température et la pluie**, avec une **croissance forte entre 2011 et 2012** : on privilégie donc des modèles non linéaires (arbres, boosting) avec une variable d'interaction `workingday × hr` et une validation strictement chronologique.# Résultats et décisions de l'EDA — Bike Sharing (`hour.csv`)

**Cible :** `cnt` (locations de vélos par heure) — 17 379 lignes × 17 colonnes.

> Les valeurs lues sur les graphiques sont **approximatives**. Les points de l'EDA dont les résultats n'ont pas encore été fournis sont listés en section 6.

---

## 1. Matrice de corrélation (Pearson)

### Résultats

| Variable | Corrélation avec `cnt` | Lecture |
|---|---|---|
| `temp` | 0,40 | Plus il fait chaud, plus la demande augmente |
| `atemp` | 0,40 | Identique à `temp` |
| `hr` | 0,39 | Lien partiel : la relation réelle est non linéaire (deux pics) |
| `hum` | −0,32 | Plus l'humidité est élevée, plus la demande baisse |
| `yr` | 0,25 | Effet de croissance 2011 → 2012 |
| `season` | 0,18 | Effet saisonnier modéré |
| `weathersit` | −0,14 | Mauvais temps, moins de locations |
| `mnth` | 0,12 | Faible |
| `windspeed` | 0,09 | Faible |
| `weekday`, `workingday`, `holiday` | ≈ 0,03 | Pas de lien linéaire direct (l'effet passe par la forme de la courbe horaire, voir section 3) |

### Multicolinéarité

| Paire | Corrélation | Constat |
|---|---|---|
| `temp` / `atemp` | 0,99 | Quasi identiques |
| `mnth` / `season` | 0,83 | Redondantes, peu gênant pour les arbres |
| `hum` / `weathersit` | 0,42 | Logique : plus d'humidité, plus de pluie |
| `yr` / variables météo | ≈ 0 | `yr` capte une croissance pure, indépendante de la météo |

---

## 2. Distribution de `cnt`

### Résultats

- Forte **asymétrie à droite** : beaucoup de petites valeurs (près de 3 000 heures dans la première classe) et une longue queue jusqu'à environ 1 000.
- **Médiane** autour de 140 locations par heure.
- Le boxplot signale des outliers au-delà d'environ 640. Ce ne sont **pas des erreurs** : ce sont les heures de pointe (section 3).
- `log1p(cnt)` resserre la queue mais reste irrégulier : un plateau entre 2 et 3,5 (régime de nuit, faible demande) et un pic vers 5,4 (régime de jour).

---

## 3. Profil horaire selon le type de jour

### Résultats

| Type de jour | Forme du profil | Valeurs approximatives |
|---|---|---|
| **Jour ouvré** | Deux pics nets : matin et soir (trajets domicile-travail) | ≈ 475 à 8h, ≈ 525 à 17h, ≈ 490 à 18h |
| **Week-end** | Grande bosse en milieu de journée | ≈ 380 entre 12h et 15h |
| **Férié** | Allure proche du week-end, niveau plus bas, petit relief à 8h | ≈ 300 au maximum |

- La nuit (0h à 3h), le week-end est plus actif que les jours ouvrés.
- L'effet de l'heure **dépend du type de jour** : c'est le graphique le plus important de l'EDA.

---

## 4. Météo et tendance (heures de pointe, jours ouvrés)

### Résultats

| `weathersit` | Médiane de `cnt` aux heures de pointe |
|---|---|
| 1 (clair) | ≈ 540 |
| 2 (brume / nuageux) | ≈ 455 |
| 3 (pluie / neige légère) | ≈ 235 |
| 4 (conditions extrêmes) | 1 seul point observé |

- Une pluie légère **divise la demande par plus de deux**.
- **Température :** la demande monte avec `temp` jusqu'à environ 0,7-0,8 (soit 29 à 33 °C), puis plafonne. Effet **non linéaire**.
- **Année :** à température égale, 2012 est nettement au-dessus de 2011.
- Forte dispersion à température fixe : l'heure et la météo expliquent une grande part du reste.

---

## 5. Décisions pour les phases suivantes

| # | Décision | Raison | Phase |
|---|---|---|---|
| 1 | **Garder tous les outliers** de `cnt` | Ce sont les heures de pointe, justement à prédire | 3 |
| 2 | **Tester `log1p(cnt)`** comme cible | Réduit l'asymétrie ; à comparer avec la cible brute (surtout modèles linéaires et RMSLE) | 5 |
| 3 | **Supprimer `atemp`**, garder `temp` | Corrélation de 0,99 | 3 |
| 4 | **Supprimer `casual`, `registered`, `instant`** | Fuite de données (`cnt = casual + registered`) et identifiant | 3 |
| 5 | **Créer `workingday × hr`** (ou une variable `day_type` : ouvré / week-end / férié) | Les profils horaires diffèrent fortement | 4 |
| 6 | **Encoder `hr` de façon cyclique** (sin/cos), éventuellement en plus de la valeur brute pour les arbres | Relation non linéaire, 23h proche de 0h | 4 |
| 7 | **Fusionner `weathersit = 4` avec `3`** | Un seul point : trop rare pour être appris | 3 |
| 8 | **Conserver `hum` et `weathersit`** malgré leur corrélation de 0,42 | Corrélation modérée, apport propre de chacune | 4 |
| 9 | **Utiliser des modèles à base d'arbres / boosting** en priorité | Relations non linéaires (heure, température) et interactions | 5 |
| 10 | **Gérer explicitement la tendance 2011 → 2012** | `yr` pèse 0,25 sur `cnt` ; les arbres ne savent pas extrapoler, donc le découpage train/test doit en tenir compte | 3 / 6 |

---

## 6. Points de l'EDA restant à analyser

Résultats non encore fournis :

| Point | Ce qu'on veut savoir | Décision attendue |
|---|---|---|
| Types, valeurs manquantes, doublons, cohérence des dates | Nombre de lignes incohérentes par test | Corrections éventuelles (Phase 3) |
| Continuité temporelle | Nombre d'heures et de jours absents, plus longs trous | Reconstruction de la grille horaire avant les lags |
| `windspeed = 0` et `hum = 0` | Nombre de valeurs suspectes | Imputation ou conservation |
| Demande selon `hr`, `weekday`, `mnth`, `season`, `yr`, `holiday` | Graphiques en barres | Vérifier notamment le sens de `season` (1 = hiver ?) |
| `casual` vs `registered` | Profils et sensibilité à la météo | Compréhension uniquement |
| Croissance 2011 → 2012 | Pourcentage de croissance par mois, évolution du profil horaire | Choix de la stratégie de validation et de la variable de tendance |

---

## 7. Résumé en une phrase

La demande est surtout structurée par **l'heure et le type de jour** (deux pics les jours ouvrés, une bosse le week-end), modulée par **la température et la pluie**, avec une **croissance forte entre 2011 et 2012** : on privilégie donc des modèles non linéaires (arbres, boosting) avec une variable d'interaction `workingday × hr` et une validation strictement chronologique.