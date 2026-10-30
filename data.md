# Dictionnaire des données — `hour.csv` (Bike Sharing)

17 379 lignes × 17 colonnes — une ligne par heure (2011-2012). **Cible : `cnt`.**

## 1. Dictionnaire

| Colonne | Signification | Valeurs / unité | Exemple (ligne 1) | Rôle dans le projet |
|---|---|---|---|---|
| `instant` | Numéro de la ligne (identifiant) | Entier croissant | 1 | À supprimer |
| `dteday` | Date du jour | AAAA-MM-JJ | 2011-01-01 | Feature |
| `season` | Saison | 1 à 4 (attention : 1 correspond à l'hiver dans les données, janvier = 1) | 1 | Feature |
| `yr` | Année | 0 = 2011, 1 = 2012 | 0 → 2011 | Feature |
| `mnth` | Mois | 1 à 12 | 1 → janvier | Feature |
| `hr` | Heure de la journée | 0 à 23 | 0 → minuit | Feature |
| `holiday` | Jour férié | 1 = férié, 0 = non | 0 | Feature |
| `weekday` | Jour de la semaine | 0 = dimanche, 1 = lundi, …, 6 = samedi | 6 → samedi | Feature |
| `workingday` | Jour ouvré (ni week-end ni férié) | 1 = oui, 0 = non | 0 (samedi) | Feature |
| `weathersit` | Situation météo | 1 = clair ; 2 = brume/nuageux ; 3 = pluie/neige légère ; 4 = conditions extrêmes (très rare) | 1 → ciel clair | Feature |
| `temp` | Température normalisée | Valeur réelle ÷ 41 (°C) | 0.24 → ≈ 9,8 °C | Feature |
| `atemp` | Température ressentie normalisée | Valeur réelle ÷ 50 (°C) | 0.2879 → ≈ 14,4 °C | Redondante |
| `hum` | Humidité normalisée | Valeur réelle ÷ 100 (%) | 0.81 → 81 % | Feature |
| `windspeed` | Vitesse du vent normalisée | Valeur réelle ÷ 67 ; 0 = vent nul ou valeur manquante | 0.0 | Feature |
| `casual` | Locations par utilisateurs occasionnels (sans abonnement) | Entier | 3 | À exclure |
| `registered` | Locations par abonnés | Entier | 13 | À exclure |
| `cnt` | Total des locations sur l'heure (`casual + registered`) | Entier | 16 | **Cible** |

**Légende du rôle :**
- **Cible** : variable à prédire.
- **Feature** : variable explicative.
- **Redondante** : quasi identique à `temp`, on n'en gardera probablement qu'une.
- **À exclure** : fuite de données (`cnt = casual + registered`), utilisée pour la compréhension uniquement.
- **À supprimer** : identifiant sans valeur prédictive.

## 2. Extrait des 4 premières lignes (unités réelles)

Facteurs de normalisation : `temp` × 41, `atemp` × 50, `hum` × 100, `windspeed` × 67.

| instant | dteday | hr | Jour | Temp. réelle (°C) | Temp. ressentie (°C) | Humidité (%) | Vent (réel) | casual | registered | cnt | Contrôle (cnt − casual − registered) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2011-01-01 | 0 | Samedi | 9,84 | 14,40 | 81 | 0 | 3 | 13 | 16 | 0 |
| 2 | 2011-01-01 | 1 | Samedi | 9,02 | 13,64 | 80 | 0 | 8 | 32 | 40 | 0 |
| 3 | 2011-01-01 | 2 | Samedi | 9,02 | 13,64 | 80 | 0 | 5 | 27 | 32 | 0 |
| 4 | 2011-01-01 | 3 | Samedi | 9,84 | 14,40 | 75 | 0 | 3 | 10 | 13 | 0 |

**Lecture :** nuit du samedi 1er janvier 2011 (minuit à 3h), environ 10 °C, humidité de 75 à 81 %, 13 à 40 locations par heure, surtout des abonnés. Le contrôle vaut 0 sur chaque ligne : `cnt = casual + registered`, d'où l'exclusion de `casual` et `registered` des features.

## 3. Points à retenir

| Sujet | Constat | Action prévue |
|---|---|---|
| `casual` / `registered` | Leur somme donne directement `cnt`. Les utiliser comme variables d'entrée reviendrait à tricher. | Compréhension uniquement ; exclure des features (Phase 3). |
| `season` | La documentation dit « 1 = printemps » mais janvier est en `season = 1` : en pratique 1 correspond à l'hiver. | Vérifier en EDA (`cnt` moyen par saison). |
| `windspeed = 0` | Présent sur les 4 lignes et très fréquent dans le dataset : probablement des valeurs manquantes encodées par 0. | Compter les zéros (EDA) puis décider du traitement (Phase 3). |
| `temp` et `atemp` | Quasi identiques (forte multicolinéarité). | Garder probablement une seule des deux. |
| Variables normalisées | `temp`, `atemp`, `hum`, `windspeed` sont déjà entre 0 et 1. | Pas de mise à l'échelle nécessaire pour les arbres ; multiplier par 41 / 50 / 100 / 67 pour interpréter. |
| `instant` | Simple numéro de ligne. | Supprimer. |