# Expériences — M1-B1 Pyrenex Crédit (Lending Club)

> Trace tes runs au fur et à mesure. Format imposé : un bloc par run, avec
> date, modèle, hyperparams, métriques, verdict.
> Commit à chaque run final (pas à chaque essai jetable).

---

## exp_001 — RF par défaut

- **Date** : YYYY-MM-DD HH:MM
- **Modèle** : RandomForestClassifier (sklearn X.Y.Z)
- **Dataset** : lending_club_train.csv (sha256 …), n=…
- **Split** : test_size=0.2, stratify=y, random_state=42
- **Hyperparamètres** : tous par défaut, `n_jobs=-1`, `random_state=42`
- **Pré-traitement** : OneHotEncoder + StandardScaler (Pipeline scikit-learn)
- **Métriques (test interne)** :
  - F1 macro : …
  - F1 défaut : …
  - ROC-AUC : …
  - Recall défaut : …
- **Métriques (holdout)** : …
- **Temps d'entraînement** : … s
- **Verdict** : …

---

## exp_002 — RF balanced (TODO — remplis avec ta config)

- **Date** :
- **Modèle** :
- **Hyperparamètres** :
- **Pré-traitement** :
- **Métriques (test interne)** :
- **Métriques (holdout)** :
- **Temps d'entraînement** :
- **Verdict** :

---

## exp_003 — (TODO — ta variante ou mission étoile ⭐ si tu y vas)

- ...
