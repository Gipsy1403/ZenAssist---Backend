import pickle

# =========================
# LOAD FILES
# =========================

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# =========================
# SAMPLE TEXT
# =========================

sample_text = [
    "My credit card payment was rejected"
]

# =========================
# TRANSFORM
# =========================

X_new = vectorizer.transform(sample_text)

# =========================
# PREDICT
# =========================

prediction = model.predict(X_new)

# =========================
# DECODE LABEL
# =========================

predicted_label = label_encoder.inverse_transform(prediction)

print("Prediction :", predicted_label[0])