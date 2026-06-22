# Présentation de soutenance — Juste des Ventilateurs

> **Usage** : ce fichier est une **source de contenu** prête à coller dans Claude
> Design (ou Gamma) pour générer une présentation. Chaque `##` = une diapo ;
> les puces = le contenu ; *« Visuel »* suggère l'illustration.

---

## 1 · Juste des Ventilateurs
- Maintenance prédictive & régulation thermique d'un datacenter (simulé via **jumeaux-chauds**)
- M2 Data/IA — LaPlateforme_
- *« Quand ça chauffe et que les capteurs s'agitent, l'IA refroidit et fait baisser la pression. »*
- *Visuel* : logo + dashboard NOC en fond

## 2 · Le problème
- Dans un datacenter : maintenir les machines en **zone sûre** sans exploser la **facture énergétique**
- Pilotage actuel = règles statiques (seuils fixes) → surchauffes, modes dégradés, **arrêts thermiques**
- Tension permanente : **sécurité thermique ↔ sobriété énergétique**
- *Visuel* : balance température/€

## 3 · La mission
- **Anticiper** les pannes par ML (préavis utile)
- **Piloter** les ventilateurs vers le meilleur compromis sécurité/énergie
- **Évaluer** en boucle fermée, chiffres à l'appui : *réduit-on les incidents ? combien de shutdowns évités ? quel impact énergie ?*
- *Visuel* : 3 icônes (anticiper / piloter / évaluer)

## 4 · Architecture
- **jumeaux-chauds** (jumeau numérique) : MQTT `:1883` (télémétrie) + REST `:8000` (pilotage fans)
- Notre service : `ingest` → `features` → `models` → `evaluation` → `supervisor` → `dashboard`
- Lecture-via-MQTT, commande-via-REST ; on ne simule jamais la physique
- *Visuel* : schéma de flux (cluster → MQTT → features → modèle → décision → PUT fan_speed)

## 5 · Les données
- **6 épisodes** collectés en direct (stress ×2, heatwave ×2, busy_weeks, nominal)
- **202 043 lignes**, 5 machines (2 masters, 3 workers)
- **9 420 pannes** `failure_60s` (4,7 %)
- Split **temporel 70/15/15 par épisode** + **embargo 60 s** → pas de fuite
- *Visuel* : tableau épisodes + barre de classes

## 6 · Feature engineering (43 features)
- **Temporelles** : dérivées T° (5/15/30 s), moyennes glissantes, volatilité, marge au seuil
- **Contextuelles** : durée zone chaude, compteurs incidents, pannes actives, statut
- **Énergétiques** : puissance fans (loi cubique P∝RPM³), PUE, énergie cumulée
- Labels *forward-looking* : `failure_60s/30s`, `hot_30s`, oracle de contrôle `action_class`
- *Visuel* : 3 colonnes de features

## 7 · Modèle de prédiction de pannes
- 4 modèles comparés : baseline seuils, **régression logistique**, random forest, gradient boosting
- Résultats (test, `failure_60s`) : GB **Recall 0.998 / PR-AUC 0.992**, Logistic **0.975 / 0.993**, RF **0.998**
- Choix production : **régression logistique** (calibrée, rapide, calculable en ligne)
- *Visuel* : courbes PR / barres Recall-F1-PR_AUC

## 8 · Explicabilité (atout clé)
- La régression logistique est **linéaire** → contribution exacte de chaque feature
- `contribution = coefficient × feature standardisée` (rouge = pousse vers la panne, vert = rassure)
- Affichée **par machine** dans le dashboard, en temps réel
- *Visuel* : barres divergentes d'explicabilité d'une machine

## 9 · Contrôleur de ventilation
- Baselines : RPM fixe, seuils, **PID**
- **Supervisé** : classifieur qui reproduit l'oracle (consigne RPM optimale)
- **Score multi-objectif** : α·risque + β·chaleur + γ·énergie
- Niveaux discrets : `{800, 1500, 2500, 3500, 4500}` RPM
- *Visuel* : table contrôleurs

## 10 · Évaluation en boucle fermée — LE résultat
- On pilote **réellement** le simulateur (≠ rejeu offline) → impact **causal**
- Scénario heatwave, sims fraîches par contrôleur :

| Politique | Arrêts | Évités | T_max | kWh fans |
|---|---|---|---|---|
| Sans refroidissement | **4** | — | 87,9 | 0,9 |
| Natif (auto) | 0 | 4 | 86,3 | 1,4 |
| **Supervisé (ML)** | **0** | **4** | **72,0** | **0,34** |
| Plein régime 4500 | 0 | 4 | 57,9 | 157 |

- **Le ML évite les 4 pannes évitables, 16 °C de marge, ~460× moins d'énergie que le plein régime**
- *Visuel* : graphe barres (arrêts) + ligne (T_max), seuil 88 °C

## 11 · Intégrité méthodologique (rigueur)
- **Pas de fuite train/test** : scaler sur train, réglages sur val, métriques sur test isolé, features causales
- **Circularité détectée** : `is_degraded` corrèle +0,63 ; 74 % des positifs sont **déjà** en panne
- Correctifs : **métriques anticipatoires** (machine encore saine), **embargo**, option « anticipation pure »
- Chiffre honnête : logistique **Recall 0.90 / Precision 0.83** ; RF robuste à 0.99 sans features de statut
- Préavis réel **~14 s** (pannes *fast-onset*) — limite assumée, documentée
- *Visuel* : tableau « annoncé vs anticipatoire »

## 12 · Poste de pilotage (démo live)
- Dashboard NOC autonome (zéro dépendance) branché sur jumeaux-chauds
- Serveurs rackmount, **ventilateurs animés** (2/machine), **risque ML + explicabilité**, **consigne IA**
- **Pilotage actif** : boutons RPM/mode + **AUTO-PILOTE** (boucle de décision live)
- Démontré : auto-pilote ON → fans montent → **températures chutent**
- *Visuel* : capture du dashboard (mode LIVE)

## 13 · Résultats clés
- **4 shutdowns évitables évités** par le ML (preuve causale)
- **Sécurité + sobriété** simultanées (72 °C @ 0,34 kWh)
- Prédiction honnête : **Recall anticipatoire 0.90**
- **221 tests** automatisés verts
- *Visuel* : 4 chiffres en gros

## 14 · Limites & perspectives
- Préavis court (pannes fast-onset) → features précurseurs / horizon plus long
- Équité boucle fermée : intégrer un reset complet (Phase 9bis)
- MLOps : CI/CD, monitoring de dérive
- Pilotage : garde-fous, RL léger, multi-rack
- *Visuel* : roadmap

## 15 · Conclusion
- Un service complet : ingestion → ML → contrôle → **boucle fermée** → **pilotage live**
- Démontre, chiffres à l'appui, le gain d'une supervision IA : **moins d'incidents, moins d'énergie**
- Démarche **rigoureuse et honnête** (audit d'intégrité, limites assumées)
- *Visuel* : schéma récap + remerciements
