from src import load_data, build_text_column, vectorize_text
from src import build_models
from src import RAW_CSV
from sklearn.model_selection import train_test_split

def train_final_model():
    # Load full dataset for final training
    df = load_data(RAW_CSV)
    df = build_text_column(df)

    X = df["text"]
    y = df["fraudulent"]

    # Train test split for vectorizer initialization
    X_train_text, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit vectorizer on all training text
    X_train, _, vectorizer = vectorize_text(X_train_text, X_train_text)

    # Select the best model (Linear SVM for now)
    model = build_models()["LinearSVM"]
    model.fit(X_train, y_train)

    return model, vectorizer

def predict_posting(text):
    model, vectorizer = train_final_model()
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    label = "FAKE JOB" if prediction == 1 else "REAL JOB"
    return label

if __name__ == "__main__":
    sample_job = input("Paste a job posting description: ")
    result = predict_posting(sample_job)
    print("\nPrediction:", result)
