import joblib
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score,  auc
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


# Logs will be saved in 'logs' directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Logging into text file
class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, message):
        for f in self.files:
            f.write(message)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

log_filename = os.path.join(
    log_dir,
    f"model_training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
)

log_file = open(log_filename, "w", encoding="utf-8")

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Logging started: {log_filename}")

# Import Dataset
df = pd.read_csv('../EDA_&_Pre-processing/preprocessed_data.csv')

# Ignore future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Dictionaries
results = {}
predictions = {}
selected_model = {}
tuned_results = {}

# The 3 Machine Learning Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=30),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=30)
}

# 2.2 Supervised Text Classification Model
# x = Features (text data)
# y = Target variable (sentiment labels)
x_content = df['content']
y_sentiment = df['sentiment']

# Dataset Splitting
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    x_content, y_sentiment, train_size=0.8, test_size=0.2, random_state=30, stratify=y_sentiment
)

# Create TF-IDF Vectorizer and perform transformation train & test data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Evaluate each model accuracy
for name, model in models.items():
    print("\n" + "="*60)
    print(f"Training Model: {name}")
    
    # Train model
    model.fit(X_train_vec, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_vec)
    predictions[name] = y_pred
    
    # Calculate model accuracy
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    
    # Model performance report
    print("Model Accuracy:", round(acc, 4))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Summary of all model accuracies
print("="*60)
print("Summary of Model Accuracies:\n")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

# 2.3 Hyper Parameter Tuning
# Define parameter grids
param_grid = {
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    "Naive Bayes": {
        'alpha': [0.1, 0.5, 1.0]
    },
    "Linear SVM": {
        'C': [0.1, 1, 10]
    }
}

# Perform Grid Search for each model
for name, model in models.items():
    print("\n" + "="*60)
    print(f"Tuning Model: {name}")

    # Grid Search setup
    grid = GridSearchCV(
        model,
        param_grid[name],
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit Grid Search
    grid.fit(X_train_vec, y_train)

    # Display best parameters
    selected_model[name] = grid.best_estimator_
    print("Best Parameters:", grid.best_params_)

    # Evaluate tuned model accuracy
    y_pred = grid.predict(X_test_vec)
    tuned_acc = accuracy_score(y_test, y_pred)
    tuned_results[name] = tuned_acc

    print("Tuned Accuracy:", round(tuned_acc, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

# Summary of baseline vs tuned results of all models
print("\n" + "="*70)
print("Summary of Baseline vs Tuned Models:\n")

for name in models.keys():
    print(f"{name}:\nBaseline={results[name]:.4f}\nTuned={tuned_results[name]:.4f}\n")

# Binarize the labels (works for multiclass)
classes = sorted(y_sentiment.unique())
y_test_bin = label_binarize(y_test, classes=classes)
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))

for name, model in selected_model.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_vec)
    else:  # For LinearSVC
        y_score = model.decision_function(X_test_vec)
    
    # Compute micro-average ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    # Plot the curve
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

# Diagonal line for random classifier
plt.plot([0, 1], [0, 1], 'r--', lw=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Select the best tuned model based on accuracy
best_tuned = max(tuned_results, key=tuned_results.get)
print("\n" + "="*60)
print("Selected Model:\n")
print("Model:", best_tuned)
print("Accuracy:", round(tuned_results[best_tuned], 4))

# Export vectorizer and the selected model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(selected_model[best_tuned], "model.pkl")

print("\nModel and vectorizer have been successfully saved.")

