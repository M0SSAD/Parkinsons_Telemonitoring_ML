import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve


def get_models():
    return {
        'SVM (RBF)': SVC(kernel='rbf', C=50.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, C=1.0, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', max_depth=10, n_estimators=100),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance')
    }


def train_and_evaluate(models, X_train_scaled, X_train_poly, y_train_balanced, X_test_scaled, X_test_poly, y_test):
    results = []
    trained_models = {}

    for name, model in models.items():
        if name == 'Logistic Regression':
            model.fit(X_train_poly, y_train_balanced)
            X_eval = X_test_poly
        else:
            model.fit(X_train_scaled, y_train_balanced)
            X_eval = X_test_scaled

        y_probs_test = model.predict_proba(X_eval)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs_test)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        y_pred = np.where(y_probs_test >= optimal_threshold, 1, 0)

        trained_models[name] = {
            'model': model,
            'y_pred': y_pred,
            'y_probs': y_probs_test,
            'threshold': optimal_threshold
        }

    return trained_models
