import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import os
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score, make_scorer
import warnings

warnings.simplefilter('ignore')

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)

# Налаштування
data = pd.read_csv('C:\\Users\\wital\\smart_sys5\\data\\train.csv')

X = data.drop(columns=['Unnamed: 0', 'Target_Graduate', 'Target_Enrolled'])
y = np.where((data['Target_Graduate'] == 1) | (data['Target_Enrolled'] == 1), 0, 1)  # 0 - не відрахований, 1 - відрахований
X_columns = X.columns.tolist()

old_class_counts = pd.Series(y).value_counts()
print(old_class_counts)

y = np.select(
    [data['Target_Graduate'] == 1, data['Target_Enrolled'] == 1],
    [0, 1],  # 0 - Graduate, 1 - Enrolled
    default=2  # 2 - Dropout, якщо обидва поля рівні 0
)

old_class_counts = pd.Series(y).value_counts()
print("Кількість зразків у початкових класах:\n", old_class_counts)

# Генерування зразків за допомогою SMOTE
X = data.drop(columns=['Unnamed: 0', 'Target_Graduate', 'Target_Enrolled'])
X_columns = X.columns.tolist()

smote = SMOTE(sampling_strategy='auto', random_state=42)
X, y = smote.fit_resample(X, y)

new_class_counts = pd.Series(y).value_counts()
print("Кількість зразків у збалансованих класах:\n", new_class_counts)

X_train = data.drop(columns=['Unnamed: 0', 'Target_Graduate', 'Target_Enrolled'])
y_train = np.where((data['Target_Graduate'] == 1) | (data['Target_Enrolled'] == 1), 0, 1)  # 0 - не відрахований, 1 - відрахований

print(X_train.shape)
print(y_train.shape)

# Сітки гіперпараметрів кожної моделі
rf_params = {
    'n_estimators': [100, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True],
    'class_weight': ['balanced', None]
}

gb_params = {
    'n_estimators': [100, 300],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'subsample': [0.8, 1.0],
    'max_features': ['sqrt', 'log2']
}

lr_params = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs'],
    'class_weight': ['balanced', None],
    'max_iter': [5000]
}

os.makedirs('C:\\Users\\wital\\smart_sys5\\models\\RandomForest', exist_ok=True)
os.makedirs('C:\\Users\\wital\\smart_sys5\\models\\GradientBoosting', exist_ok=True)
os.makedirs('C:\\Users\\wital\\smart_sys5\\models\\LogisticRegression', exist_ok=True)

def save_feature_importances(model, feature_names, filename):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = abs(model.coef_[0])
    else:
        return  # якщо модель не підтримує важливість ознак

    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    feature_importances.to_csv(filename, index=False)

log_file = 'C:\\Users\\wital\\smart_sys5\\models\\training_logs.txt'
with open(log_file, 'w') as f:
    f.write("Log of hyperparameter tuning and training:\n\n")

recall_scorer = make_scorer(recall_score, average='macro')

# Словники для збереження моделей і гіперпараметрів
models = {
    'RandomForest': (RandomForestClassifier(), rf_params, 'C:\\Users\\wital\\smart_sys5\\models\\RandomForest'),
    'GradientBoosting': (GradientBoostingClassifier(), gb_params, 'C:\\Users\\wital\\smart_sys5\\models\\GradientBoosting'),
    'LogisticRegression': (LogisticRegression(), lr_params, 'C:\\Users\\wital\\smart_sys5\\models\\LogisticRegression')
}

# Підбір гіперпараметрів для кожної моделі
for model_name, (model, params, model_dir) in models.items():
    print(f"Починаю підбір для {model_name}...")

    grid_search = GridSearchCV(estimator=model, param_grid=params, scoring=recall_scorer, cv=5, verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)


    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv(f"{model_dir}/grid_search_results.csv", index=False)

    # Логування найкращих параметрів і результатів
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    with open(log_file, 'a') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Best Recall Score: {best_score}\n")
        f.write("=" * 50 + "\n")

    model_path = os.path.join(model_dir, f"{model_name}_best_model.joblib")
    joblib.dump(best_model, model_path)

    feature_importances_path = os.path.join(model_dir, "feature_importances.csv")
    save_feature_importances(best_model, X_columns, feature_importances_path)

print("Гіперпараметри підібрано та моделі збережено.")
