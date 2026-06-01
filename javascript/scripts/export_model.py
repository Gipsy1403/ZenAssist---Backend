# =====================================================
# IMPORTS
# =====================================================

import os
import sys
import json
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


# =====================================================
# VERSION DU MODELE
# =====================================================

# Récupère le tag GitHub ou une version par défaut
version = sys.argv[1] if len(sys.argv) > 1 else "v0.0.0"

# Sécurise le nom de version
version = version.replace("/", "_")

print(f"🚀 Model version: {version}")


# =====================================================
# CHARGEMENT DES DONNEES
# =====================================================

df = pd.read_csv("javascript/datasets/dataset_sample.csv")

print(f"✅ Dataset chargé : {len(df)} lignes")


# =====================================================
# NORMALISATION DES ENTREPRISES
# =====================================================

# Nettoyage des noms de compagnies
df["Company_clean"] = (
    df["Company"]
    .fillna("")
    .str.lower()
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
)

# Création d'un mapping pour garder
# la version la plus fréquente du nom
mapping = {}

grouped = df.groupby("Company_clean")["Company"]

for name, variants in grouped:
    mapping[name] = variants.value_counts().idxmax()

# Nom final normalisé
df["Company_final"] = df["Company_clean"].map(mapping)


# =====================================================
# NETTOYAGE DES TAGS
# =====================================================

def nettoyer_tags(dataframe):
    """
    Fusionne certaines anciennes catégories CFPB
    pour éviter les doublons de labels.
    """

    mapping_tags = {
        "Credit reporting":
            "Credit reporting, credit repair services, or other personal consumer reports",

        "Credit card":
            "Credit card or prepaid card",

        "Prepaid card":
            "Credit card or prepaid card",

        "Payday loan":
            "Payday loan, title loan, or personal loan",

        "Virtual currency":
            "Money transfer, virtual currency, or money service"
    }

    dataframe["Tag"] = dataframe["Tag"].replace(mapping_tags)

    return dataframe


df = nettoyer_tags(df)


# =====================================================
# SUPPRESSION DES VALEURS VIDES
# =====================================================

# On garde uniquement les réclamations existantes
df = df[df["Consumer Claim"].notna()].copy()

print(f"✅ Dataset nettoyé : {len(df)} lignes")


# =====================================================
# PREPARATION DU TEXTE
# =====================================================

# Création du texte final utilisé par le modèle
df["text"] = (
    df["Consumer Claim"].fillna("")
    + " "
    + df["Company_final"].fillna("")
).str.lower()


# =====================================================
# LABEL ENCODING
# =====================================================

le = LabelEncoder()

df["label"] = le.fit_transform(df["Tag"])

print(f"✅ Nombre de classes : {len(le.classes_)}")


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("✅ Séparation train/test effectuée")


# =====================================================
# TF-IDF VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 3),
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("✅ Vectorisation terminée")


# =====================================================
# ENTRAINEMENT DU MODELE
# =====================================================

model = LinearSVC(
    random_state=42,
    class_weight="balanced",
    C=0.5
)

model.fit(X_train_vec, y_train)

print("✅ Modèle entraîné")


# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test_vec)


# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

print(f"🎯 Accuracy : {accuracy:.4f}")


# =====================================================
# METRICS
# =====================================================

metrics = {
    "version": version,
    "accuracy": float(accuracy),
    "labels": {
        int(i): label
        for i, label in enumerate(le.classes_)
    },
    "classification_report": report
}


# =====================================================
# EXPORT DOSSIER
# =====================================================

os.makedirs("models", exist_ok=True)


# =====================================================
# FONCTION DE SAUVEGARDE
# =====================================================

def save_pickle(obj, path):
    """
    Sauvegarde un objet Python au format pickle.
    """

    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =====================================================
# EXPORT MODEL
# =====================================================

save_pickle(
    model,
    f"models/{version}_model.pkl"
)

save_pickle(
    vectorizer,
    f"models/{version}_vectorizer.pkl"
)

save_pickle(
    le,
    f"models/{version}_label_encoder.pkl"
)


# =====================================================
# EXPORT METRICS
# =====================================================

with open(f"models/{version}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)


print("✅ Export terminé")