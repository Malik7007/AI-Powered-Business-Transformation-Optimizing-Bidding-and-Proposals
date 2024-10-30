import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint

# Load the data
df = pd.read_csv("Dataset-Go-NoGo selected Features.csv", encoding='latin1')

# Preprocess categorical features
categorical_columns = [
    "Project Scope Complexity",
    "Technical Feasibility",
    "Financial Feasibility",
    "Contract Terms",
    "Relevant Experience",
    "Internal Resources Available",
    "Risk Level",
    "Market Demand",
    "Strategic Alignment",
    "Long Term Impact" 
]

def preprocess_data(df, categorical_columns):
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    return df, le_dict

def save_preprocessing_objects(le_dict, scaler, file_paths):
    with open(file_paths["label_encoders"], "wb") as f:
        pickle.dump(le_dict, f)
    with open(file_paths["scaler"], "wb") as f:
        pickle.dump(scaler, f)

def load_preprocessing_objects(file_paths):
    with open(file_paths["label_encoders"], "rb") as f:
        le_dict = pickle.load(f)
    with open(file_paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    return le_dict, scaler

def train_and_save_models(X_train, y_train, file_paths):
    # Define model parameter grids
    param_dist_rf = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10)
    }
    param_grid_lr = {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    param_grid_svc = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    param_grid_knn = {
        'n_neighbors': [3, 5, 11, 19],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    param_grid_mlp = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']
    }
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9]
    }

    # Train and save RandomForestClassifier
    rf_classifier = RandomForestClassifier()
    grid_search_rf = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist_rf, n_iter=12, cv=5, verbose=1, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    best_model_rf = grid_search_rf.best_estimator_
    with open(file_paths["best_model_rf"], "wb") as f:
        pickle.dump(best_model_rf, f)

    # Train and save LogisticRegression
    grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5)
    grid_search_lr.fit(X_train, y_train)
    best_model_lr = grid_search_lr.best_estimator_
    with open(file_paths["best_model_lr"], "wb") as f:
        pickle.dump(best_model_lr, f)

    # Train and save SVM
    grid_search_svc = GridSearchCV(SVC(), param_grid_svc, cv=5)
    grid_search_svc.fit(X_train, y_train)
    best_model_svc = grid_search_svc.best_estimator_
    with open(file_paths["best_model_svc"], "wb") as f:
        pickle.dump(best_model_svc, f)

    # Train and save K-Nearest Neighbors
    grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5)
    grid_search_knn.fit(X_train, y_train)
    best_model_knn = grid_search_knn.best_estimator_
    with open(file_paths["best_model_knn"], "wb") as f:
        pickle.dump(best_model_knn, f)

    # Train and save Neural Network (MLP)
    grid_search_mlp = GridSearchCV(MLPClassifier(max_iter=1000), param_grid_mlp, cv=5)
    grid_search_mlp.fit(X_train, y_train)
    best_model_mlp = grid_search_mlp.best_estimator_
    with open(file_paths["best_model_mlp"], "wb") as f:
        pickle.dump(best_model_mlp, f)

    # Train and save Naive Bayes
    naive_bayes = GaussianNB()
    naive_bayes.fit(X_train, y_train)
    with open(file_paths["best_model_nb"], "wb") as f:
        pickle.dump(naive_bayes, f)

    # Train and save XGBoost
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search_xgb = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid_xgb, cv=5)
    grid_search_xgb.fit(X_train, y_train)
    best_model_xgb = grid_search_xgb.best_estimator_
    with open(file_paths["best_model_xgb"], "wb") as f:
        pickle.dump(best_model_xgb, f)

    return {
        "Random Forest": best_model_rf,
        "Logistic Regression": best_model_lr,
        "SVM": best_model_svc,
        "K-Nearest Neighbors": best_model_knn,
        "Neural Network": best_model_mlp,
        "Naive Bayes": naive_bayes,
        "XGBoost": best_model_xgb
    }

def evaluate_models_and_save_report(models, X_test, y_test, feature_names, report_file="model_report.txt"):
    with open(report_file, "w") as report:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred) * 100
            report_text = classification_report(y_test, y_pred)

            # Write to the report file
            report.write(f"Model: {name}\n")
            report.write(f"Accuracy: {accuracy:.2f}%\n")
            report.write("Classification Report:\n")
            report.write(report_text + "\n\n")

            # Feature importance for models that support it
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                sns.barplot(y=[feature_names[i] for i in indices[:10]], x=importances[indices[:10]])
                plt.title(f"Feature Importance for {name}")
                plt.xlabel("Importance")
                plt.ylabel("Feature")
                plt.savefig(f"{name}_feature_importance.png")
                plt.clf()
            elif hasattr(model, "coef_"):
                # Logistic Regression feature importances (absolute values of coefficients)
                importances = np.abs(model.coef_[0])
                indices = np.argsort(importances)[::-1]
                sns.barplot(y=[feature_names[i] for i in indices[:10]], x=importances[indices[:10]])
                plt.title(f"Coefficient Magnitude for {name}")
                plt.xlabel("Coefficient Magnitude")
                plt.ylabel("Feature")
                plt.savefig(f"{name}_coefficients.png")
                plt.clf()

# Load data and preprocess
df_processed, le_dict = preprocess_data(df, categorical_columns)
X = df.drop(columns=["Initial Go NoGo Decision"])
y_encoded_go_no_go = LabelEncoder().fit_transform(df["Initial Go NoGo Decision"])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_go_no_go, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and save models
file_paths_go_no_go = {
    "label_encoders": "le_dict_go_no_go.pkl",
    "scaler": "scaler_go_no_go.pkl",
    "best_model_rf": "rf_go_no_go.pkl",
    "best_model_lr": "lr_go_no_go.pkl",
    "best_model_svc": "svc_go_no_go.pkl",
    "best_model_knn": "knn_go_no_go.pkl",
    "best_model_mlp": "mlp_go_no_go.pkl",
    "best_model_nb": "nb_go_no_go.pkl",
    "best_model_xgb": "xgb_go_no_go.pkl"
}
save_preprocessing_objects(le_dict, scaler, file_paths_go_no_go)
models_go_no_go = train_and_save_models(X_train_scaled, y_train, file_paths_go_no_go)

# Evaluate and save report with feature importance
evaluate_models_and_save_report(models_go_no_go, X_test_scaled, y_test, X.columns)
