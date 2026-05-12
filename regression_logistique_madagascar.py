# =============================================================================
# PROJET : Probabilité de transition vers les énergies renouvelables
# MODÈLE  : Régression Logistique
# CAS     : Madagascar et pays africains comparables
# AUTEUR  : [Votre nom]
# DATE    : 2026
# =============================================================================

# ─────────────────────────────────────────────
# ÉTAPE 1 : Importer les bibliothèques
# (les "boîtes à outils" dont on a besoin)
# ─────────────────────────────────────────────
import pandas as pd                          # Pour lire et manipuler les données
import numpy as np                           # Pour les calculs mathématiques
import matplotlib.pyplot as plt              # Pour faire les graphiques
import matplotlib.patches as mpatches       # Pour les légendes personnalisées
import seaborn as sns                        # Pour de beaux graphiques statistiques

from sklearn.linear_model import LogisticRegression   # Le modèle de régression logistique
from sklearn.model_selection import train_test_split  # Pour diviser les données
from sklearn.metrics import (                          # Pour évaluer le modèle
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from sklearn.preprocessing import StandardScaler      # Pour normaliser les données

import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  RÉGRESSION LOGISTIQUE — TRANSITION ÉNERGÉTIQUE")
print("  Cas d'étude : Madagascar et Afrique")
print("=" * 60)


# ─────────────────────────────────────────────
# ÉTAPE 2 : Créer le jeu de données
# Sources : IRENA, Our World in Data, Banque Mondiale
# ─────────────────────────────────────────────
print("\n📊 ÉTAPE 2 : Chargement des données...")

data = {
    'pays': [
        # Madagascar (notre cas central)
        'Madagascar','Madagascar','Madagascar','Madagascar','Madagascar',
        'Madagascar','Madagascar','Madagascar','Madagascar','Madagascar',
        # Éthiopie (fort renouvelable)
        'Éthiopie','Éthiopie','Éthiopie','Éthiopie','Éthiopie',
        # Mozambique (fort renouvelable)
        'Mozambique','Mozambique','Mozambique','Mozambique','Mozambique',
        # Sénégal (faible renouvelable)
        'Sénégal','Sénégal','Sénégal','Sénégal','Sénégal',
        # Kenya (transition en cours)
        'Kenya','Kenya','Kenya','Kenya','Kenya',
        # Tanzanie (intermédiaire)
        'Tanzanie','Tanzanie','Tanzanie','Tanzanie','Tanzanie',
        # Nigeria (forte dépendance fossile)
        'Nigeria','Nigeria','Nigeria','Nigeria','Nigeria',
        # Rwanda (fort renouvelable)
        'Rwanda','Rwanda','Rwanda','Rwanda','Rwanda',
        # Burkina Faso (faible renouvelable)
        'Burkina Faso','Burkina Faso','Burkina Faso','Burkina Faso','Burkina Faso',
        # Ghana (intermédiaire)
        'Ghana','Ghana','Ghana','Ghana','Ghana',
    ],
    'annee': [
        2000,2005,2010,2014,2015,2016,2017,2018,2019,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
        2000,2005,2010,2015,2020,
    ],
    # x1 : PIB par habitant (USD) — Source : Banque Mondiale
    'gdp_par_habitant': [
        239, 289, 398, 450, 390, 400, 420, 460, 510, 480,
        110, 160, 360, 619, 936,
        213, 328, 430, 471, 494,
        462, 621, 919, 964, 1451,
        340, 550, 900, 1335, 1838,
        268, 384, 545, 877, 1080,
        378, 760, 1140, 2640, 2065,
        195, 271, 526, 718, 798,
        219, 373, 575, 652, 754,
        256, 481, 1320, 1606, 2200,
    ],
    # x2 : Part fossile dans l'électricité (%) — Source : Our World in Data
    'part_fossile': [
        38.9, 43.6, 47.7, 46.7, 45.4, 52.8, 55.8, 55.2, 60.4, 57.8,
        0.7,  0.5,  0.4,  0.2,  0.1,
        2.5,  3.8,  4.9,  9.7, 13.6,
        92.0, 90.5, 89.6, 89.5, 69.0,
        45.0, 32.0, 28.0, 25.0, 20.0,
        60.0, 66.0, 66.0, 66.0, 62.0,
        98.0, 97.5, 95.0, 92.5, 87.5,
        20.0, 18.0, 15.0, 13.0,  7.0,
        94.0, 92.5, 90.7, 91.0, 82.0,
        50.0, 52.0, 53.0, 62.0, 58.0,
    ],
    # x3 : CO2 par habitant (tonnes) — Source : Our World in Data
    'co2_par_habitant': [
        0.10, 0.10, 0.12, 0.12, 0.11, 0.11, 0.12, 0.13, 0.13, 0.09,
        0.07, 0.07, 0.08, 0.13, 0.18,
        0.10, 0.11, 0.24, 0.29, 0.31,
        0.40, 0.41, 0.48, 0.60, 0.67,
        0.23, 0.28, 0.30, 0.35, 0.39,
        0.07, 0.09, 0.17, 0.23, 0.23,
        0.52, 0.56, 0.54, 0.58, 0.55,
        0.06, 0.06, 0.06, 0.07, 0.07,
        0.08, 0.10, 0.14, 0.16, 0.19,
        0.37, 0.40, 0.45, 0.52, 0.55,
    ],
    # Variable cible Y : part d'électricité renouvelable (%)
    'renew_share': [
        61.1, 56.4, 52.3, 53.3, 54.6, 47.2, 44.2, 44.8, 39.6, 42.2,
        99.3, 99.5, 99.6, 99.8, 99.9,
        97.5, 96.2, 95.1, 90.3, 86.4,
         8.0,  9.5, 10.4, 10.5, 31.0,
        55.0, 68.0, 72.0, 75.0, 80.0,
        40.0, 34.0, 34.0, 34.0, 38.0,
         2.0,  2.5,  5.0,  7.5, 12.5,
        80.0, 82.0, 85.0, 87.0, 93.0,
         6.0,  7.5,  9.3,  9.0, 18.0,
        50.0, 48.0, 47.0, 38.0, 42.0,
    ]
}

df = pd.DataFrame(data)

# ─────────────────────────────────────────────
# ÉTAPE 3 : Créer la variable Y (0 ou 1)
# Seuil : si renouvelable >= 50% → Y = 1 (transition amorcée)
#         si renouvelable <  50% → Y = 0 (pas encore en transition)
# ─────────────────────────────────────────────
print("\n🎯 ÉTAPE 3 : Création de la variable cible Y...")

SEUIL = 50  # % de renouvelables pour considérer qu'il y a transition
df['Y'] = (df['renew_share'] >= SEUIL).astype(int)

print(f"   Seuil choisi : {SEUIL}% d'électricité renouvelable")
print(f"   Y = 1 (transition) : {df['Y'].sum()} observations")
print(f"   Y = 0 (pas encore) : {(df['Y'] == 0).sum()} observations")

# Afficher les données de Madagascar
print("\n📍 Données Madagascar :")
mdg = df[df['pays'] == 'Madagascar'][['annee','renew_share','gdp_par_habitant','part_fossile','co2_par_habitant','Y']]
print(mdg.to_string(index=False))


# ─────────────────────────────────────────────
# ÉTAPE 4 : Préparer les données pour le modèle
# ─────────────────────────────────────────────
print("\n⚙️  ÉTAPE 4 : Préparation des données...")

# Variables X (les facteurs explicatifs)
X = df[['gdp_par_habitant', 'part_fossile', 'co2_par_habitant']]
# Variable Y (ce qu'on veut prédire)
y = df['Y']

# Normalisation : met toutes les variables sur la même échelle
# (important pour que le modèle compare correctement PIB vs CO2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Division en données d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"   Données d'entraînement : {len(X_train)} observations")
print(f"   Données de test        : {len(X_test)} observations")


# ─────────────────────────────────────────────
# ÉTAPE 5 : Entraîner le modèle
# C'est ici que Python calcule A, B, C tout seul !
# ─────────────────────────────────────────────
print("\n🤖 ÉTAPE 5 : Entraînement du modèle de régression logistique...")

modele = LogisticRegression(max_iter=1000, random_state=42)
modele.fit(X_train, y_train)

# Récupérer les coefficients A, B, C calculés par Python
A = modele.coef_[0][0]   # Coefficient du PIB/habitant
B = modele.coef_[0][1]   # Coefficient de la part fossile
C = modele.coef_[0][2]   # Coefficient du CO2/habitant
intercept = modele.intercept_[0]  # La constante ε

print(f"\n   ✅ Coefficients calibrés automatiquement :")
print(f"   A (PIB/habitant)  = {A:.4f}")
print(f"   B (part fossile)  = {B:.4f}")
print(f"   C (CO2/habitant)  = {C:.4f}")
print(f"   ε (constante)     = {intercept:.4f}")
print(f"\n   Formule : X = {A:.4f}×PIB + ({B:.4f})×Fossile + {C:.4f}×CO2 + {intercept:.4f}")


# ─────────────────────────────────────────────
# ÉTAPE 6 : Évaluer le modèle
# ─────────────────────────────────────────────
print("\n📈 ÉTAPE 6 : Évaluation du modèle...")

y_pred = modele.predict(X_test)
y_proba = modele.predict_proba(X_test)[:, 1]
accuracy = modele.score(X_test, y_test)

print(f"   Précision globale : {accuracy*100:.1f}%")
print("\n   Rapport de classification :")
print(classification_report(y_test, y_pred,
      target_names=['Pas de transition (Y=0)', 'Transition (Y=1)']))


# ─────────────────────────────────────────────
# ÉTAPE 7 : Prédire pour Madagascar
# C'est le moment clé de notre cas d'étude !
# ─────────────────────────────────────────────
print("\n🇲🇬 ÉTAPE 7 : PRÉDICTION POUR MADAGASCAR...")
print("-" * 50)

# Valeurs actuelles de Madagascar (2020)
madagascar_2020 = np.array([[480, 57.8, 0.09]])
# Valeurs projetées si politiques énergétiques appliquées (objectif 85% renouvelable d'ici 2030)
madagascar_2030 = np.array([[650, 30.0, 0.08]])

# Normaliser avec le même scaler
mdg_2020_scaled = scaler.transform(madagascar_2020)
mdg_2030_scaled = scaler.transform(madagascar_2030)

# Prédictions
proba_2020 = modele.predict_proba(mdg_2020_scaled)[0][1]
proba_2030 = modele.predict_proba(mdg_2030_scaled)[0][1]

print(f"\n   Madagascar 2020 (situation actuelle) :")
print(f"   → PIB/hab = 480 USD | Fossile = 57.8% | CO2 = 0.09t")
print(f"   → Probabilité de transition = {proba_2020*100:.1f}%")

print(f"\n   Madagascar 2030 (si objectifs gouvernementaux atteints) :")
print(f"   → PIB/hab = 650 USD | Fossile = 30.0% | CO2 = 0.08t")
print(f"   → Probabilité de transition = {proba_2030*100:.1f}%")

print(f"\n   📊 Interprétation des coefficients :")
print(f"   - A = {A:.4f} : Une hausse du PIB AUGMENTE la probabilité de transition")
print(f"   - B = {B:.4f} : Une hausse des fossiles DIMINUE la probabilité de transition")
print(f"   - C = {C:.4f} : Impact du CO2 sur la transition")


# ─────────────────────────────────────────────
# ÉTAPE 8 : Graphiques
# ─────────────────────────────────────────────
print("\n📊 ÉTAPE 8 : Génération des graphiques...")

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    'Régression Logistique — Transition vers les Énergies Renouvelables\nCas d\'étude : Madagascar et Afrique',
    fontsize=14, fontweight='bold', y=0.98
)
couleurs = {'Madagascar': '#E74C3C', 'Éthiopie': '#2ECC71', 'Mozambique': '#27AE60',
            'Sénégal': '#E67E22', 'Kenya': '#3498DB', 'Tanzanie': '#9B59B6',
            'Nigeria': '#E74C3C', 'Rwanda': '#1ABC9C', 'Burkina Faso': '#F39C12', 'Ghana': '#2980B9'}

# ── Graphique 1 : Renouvelables vs PIB ──────────────────────────────
ax1 = axes[0, 0]
for pays in df['pays'].unique():
    subset = df[df['pays'] == pays]
    lw = 2.5 if pays == 'Madagascar' else 1
    alpha = 1.0 if pays == 'Madagascar' else 0.6
    mk = 'o' if pays == 'Madagascar' else 's'
    ax1.plot(subset['gdp_par_habitant'], subset['renew_share'],
             marker=mk, label=pays, color=couleurs.get(pays, 'gray'),
             linewidth=lw, alpha=alpha, markersize=5)

ax1.axhline(y=SEUIL, color='black', linestyle='--', linewidth=1, label=f'Seuil {SEUIL}%')
ax1.set_xlabel('PIB par habitant (USD)', fontsize=10)
ax1.set_ylabel('Part renouvelable (%)', fontsize=10)
ax1.set_title('x1 : PIB/habitant vs Part renouvelable', fontsize=11, fontweight='bold')
ax1.legend(fontsize=7, ncol=2)
ax1.grid(True, alpha=0.3)

# ── Graphique 2 : Coefficients A, B, C ──────────────────────────────
ax2 = axes[0, 1]
variables = ['A\n(PIB/hab)', 'B\n(Part fossile)', 'C\n(CO₂/hab)']
coefficients = [A, B, C]
bar_colors = ['#2ECC71' if v > 0 else '#E74C3C' for v in coefficients]
bars = ax2.bar(variables, coefficients, color=bar_colors, edgecolor='black', linewidth=0.5, width=0.5)
ax2.axhline(y=0, color='black', linewidth=1)
for bar, val in zip(bars, coefficients):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_title('Coefficients calibrés (A, B, C)\nVert = augmente la transition | Rouge = diminue',
              fontsize=11, fontweight='bold')
ax2.set_ylabel('Valeur du coefficient', fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# ── Graphique 3 : Courbe ROC ─────────────────────────────────────────
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
ax3.plot(fpr, tpr, color='#3498DB', lw=2, label=f'Courbe ROC (AUC = {roc_auc:.2f})')
ax3.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aléatoire')
ax3.fill_between(fpr, tpr, alpha=0.1, color='#3498DB')
ax3.set_xlabel('Taux de faux positifs', fontsize=10)
ax3.set_ylabel('Taux de vrais positifs', fontsize=10)
ax3.set_title('Courbe ROC — Performance du modèle\n(Plus l\'AUC est proche de 1, meilleur est le modèle)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ── Graphique 4 : Probabilité Madagascar 2020 vs 2030 ───────────────
ax4 = axes[1, 1]
scenarios = ['Madagascar\n2020\n(actuel)', 'Madagascar\n2030\n(objectif)']
probas = [proba_2020 * 100, proba_2030 * 100]
bar_c = ['#E74C3C' if p < 50 else '#2ECC71' for p in probas]
bars4 = ax4.bar(scenarios, probas, color=bar_c, edgecolor='black', linewidth=0.5, width=0.4)
ax4.axhline(y=50, color='black', linestyle='--', linewidth=1.5, label='Seuil 50%')
for bar, val in zip(bars4, probas):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')
ax4.set_ylim(0, 105)
ax4.set_ylabel('Probabilité de transition (%)', fontsize=10)
ax4.set_title('🇲🇬 Probabilité de transition — Madagascar\nActuel vs Objectif 2030',
              fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('resultats_regression_logistique.png', dpi=150, bbox_inches='tight')
plt.savefig('resultats_regression_logistique.pdf', bbox_inches='tight')
plt.show()

print("\n   ✅ Graphiques sauvegardés :")
print("      → resultats_regression_logistique.png")
print("      → resultats_regression_logistique.pdf")

# ─────────────────────────────────────────────
# ÉTAPE 9 : Matrice de confusion
# ─────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Prédit : Non', 'Prédit : Oui'],
            yticklabels=['Réel : Non', 'Réel : Oui'])
ax.set_title('Matrice de confusion\n(Vérifie la précision du modèle)', fontweight='bold')
plt.tight_layout()
plt.savefig('matrice_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
print("      → matrice_confusion.png")

print("\n" + "=" * 60)
print("  ✅ PROJET TERMINÉ AVEC SUCCÈS !")
print(f"  Précision du modèle : {accuracy*100:.1f}%")
print(f"  Probabilité transition Madagascar 2020 : {proba_2020*100:.1f}%")
print(f"  Probabilité transition Madagascar 2030 : {proba_2030*100:.1f}%")
print("=" * 60)
