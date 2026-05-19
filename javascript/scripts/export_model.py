# # =====================================================
# # IMPORTS
# # =====================================================
# import os
# import pickle
# import json
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import (
#     classification_report,
#     accuracy_score
# )

# # =====================================================
# # CHARGEMENT DES DONNEES
# # =====================================================

# df_sample = pd.read_csv("javascript/datasets/dataset_sample.csv")

# # =====================================================
# # Normalisation des noms d'entreprises
# # =====================================================

# df_sample["Company_clean"] = (
#     df_sample["Company"]
#     .str.lower()
#     .str.strip()
#     .str.replace(r"[^\w\s]", "", regex=True)  # enlève ponctuation
# )

# # Faire le choix de prendre la version du nom de l'entreprise le plus représenté
# mapping = {}

# grouped = df_sample.groupby("Company_clean")["Company"]

# for name, variants in grouped:
#     mapping[name] = variants.value_counts().idxmax()

# df_sample["Company_final"] = df_sample["Company_clean"].map(mapping)

# # =====================================================
# # Nettoyage des tags
# # =====================================================
# def nettoyer_tags(df):
    
#     mapping_tags = {

#         "Credit reporting":
#             "Credit reporting, credit repair services, or other personal consumer reports",

#         "Credit card":
#             "Credit card or prepaid card",

#         "Prepaid card":
#             "Credit card or prepaid card",
            
#         "Payday loan":
#             "Payday loan, title loan, or personal loan",
            
#         "Virtual currency":
#             "Money transfer, virtual currency, or money service"
#     }

#     df["Tag"] = df["Tag"].replace(mapping_tags)

#     return df

# df_sample = nettoyer_tags(df_sample)

# # =====================================================
# # NETTOYAGE DES DONNEES
# # =====================================================

# # Supprime les lignes sans plainte
# df_consumer = df_sample[df_sample["Consumer Claim"].notna()]

# # Garde uniquement les colonnes utiles
# df_new = df_consumer[
#     ["Consumer Claim", "Company_final", "Tag"]
# ]

# df_new = df_new.copy()

# # =====================================================
# # CREATION DU TEXTE
# # =====================================================

# # Fusion des colonnes texte
# df_new["text"] = (
#     df_new["Consumer Claim"].fillna("")
#     + " "
#     + df_new["Company_final"].fillna("")
# )

# # Passage en minuscules
# df_new["text"] = df_new["text"].str.lower()

# # =====================================================
# # ENCODAGE DES LABELS
# # =====================================================

# le = LabelEncoder()

# df_new["label_encoded"] = le.fit_transform(
#     df_new["Tag"]
# )

# # =====================================================
# # TRAIN / TEST SPLIT
# # =====================================================

# X = df_new["text"]
# y = df_new["label_encoded"]

# X_train, X_test, y_train, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# # =====================================================
# # VECTORIZATION
# # =====================================================

# vectorizer = TfidfVectorizer(
#     max_features=5000,
#     max_df=0.9,
#     min_df=2
# )

# X_train_vec = vectorizer.fit_transform(X_train)

# X_test_vec = vectorizer.transform(X_test)

# # =====================================================
# # MODELE
# # =====================================================

# model = LogisticRegression(max_iter=1000, random_state=42)

# model.fit(X_train_vec, y_train)

# # =====================================================
# # PREDICTIONS
# # =====================================================

# y_pred = model.predict(X_test_vec)

# # =====================================================
# # METRIQUES
# # =====================================================

# print(classification_report(y_test, y_pred))

# accuracy = accuracy_score(y_test, y_pred)

# report = classification_report(
#     y_test,
#     y_pred,
#     output_dict=True
# )

# metrics = {
#     "accuracy": accuracy,
#     "classification_report": report
# }

# # =====================================================
# # EXPORT DU MODELE
# # =====================================================

# # Créer le dossier "models" s'il n'existe pas
# os.makedirs("models", exist_ok=True)

# with open("models/model.pkl", "wb") as f:
#     pickle.dump(model, f)

# # =====================================================
# # EXPORT DU VECTORIZER
# # =====================================================

# with open("models/vectorizer.pkl", "wb") as f:
#     pickle.dump(vectorizer, f)

# # =====================================================
# # EXPORT LABEL ENCODER
# # =====================================================

# with open("models/label_encoder.pkl", "wb") as f:
#     pickle.dump(le, f)

# # =====================================================
# # EXPORT METRICS
# # =====================================================

# with open("models/metrics.json", "w") as f:
#     json.dump(metrics, f)

# print("Export terminé")

# =====================================================
# IMPORTS
# =====================================================
import os
import pickle
import json
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# VERSION DU MODELE (GitHub tag)
# =====================================================

version = sys.argv[1] if len(sys.argv) > 1 else "dev"

print(f"Model version: {version}")

# =====================================================
# CHARGEMENT DONNEES
# =====================================================

df_sample = pd.read_csv("javascript/datasets/dataset_sample.csv")

# =====================================================
# NORMALISATION COMPAGNIES
# =====================================================

df_sample["Company_clean"] = (
    df_sample["Company"]
    .str.lower()
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
)

mapping = {}
grouped = df_sample.groupby("Company_clean")["Company"]

for name, variants in grouped:
    mapping[name] = variants.value_counts().idxmax()

df_sample["Company_final"] = df_sample["Company_clean"].map(mapping)

# =====================================================
# NETTOYAGE TAGS
# =====================================================

def nettoyer_tags(df):
    mapping_tags = {
        "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit card": "Credit card or prepaid card",
        "Prepaid card": "Credit card or prepaid card",
        "Payday loan": "Payday loan, title loan, or personal loan",
        "Virtual currency": "Money transfer, virtual currency, or money service"
    }

    df["Tag"] = df["Tag"].replace(mapping_tags)
    return df

df_sample = nettoyer_tags(df_sample)

# =====================================================
# PREPARATION DATA
# =====================================================

df_consumer = df_sample[df_sample["Consumer Claim"].notna()]

df_new = df_consumer[["Consumer Claim", "Company_final", "Tag"]].copy()

df_new["text"] = (
    df_new["Consumer Claim"].fillna("")
    + " "
    + df_new["Company_final"].fillna("")
).str.lower()

# =====================================================
# LABEL ENCODING
# =====================================================

le = LabelEncoder()
df_new["label_encoded"] = le.fit_transform(df_new["Tag"])

# =====================================================
# TRAIN / TEST
# =====================================================

X = df_new["text"]
y = df_new["label_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    max_df=0.9,
    min_df=2
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =====================================================
# MODEL
# =====================================================

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

# =====================================================
# PREDICTIONS
# =====================================================

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# =====================================================
# METRICS (VERSIONNEES)
# =====================================================

metrics = {
    "version": version,
    "accuracy": accuracy,
    "classification_report": report
}

# =====================================================
# EXPORT
# =====================================================

os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Export terminé")