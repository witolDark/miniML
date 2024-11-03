import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report

# Шляхи до моделі та тестових даних
model_paths = {
    'RandomForest': 'C:\\Users\\wital\\smart_sys5\\models\\RandomForest\\RandomForest_best_model.joblib',
    'GradientBoosting': 'C:\\Users\\wital\\smart_sys5\\models\\GradientBoosting\\GradientBoosting_best_model.joblib',
    'LogisticRegression': 'C:\\Users\\wital\\smart_sys5\\models\\LogisticRegression\\LogisticRegression_best_model.joblib'
}

results_path = 'C:\\Users\\wital\\smart_sys5\\models\\results.txt'

test_data = pd.read_csv('C:\\Users\\wital\\smart_sys5\\data\\test.csv')

if test_data.empty:
    print("Тестові дані порожні, перевірте файл.")
else:
    X_test = test_data.drop(columns=['Unnamed: 0', 'Target_Graduate', 'Target_Enrolled'])
    y_test = np.where((test_data['Target_Graduate'] == 1) | (test_data['Target_Enrolled'] == 1), 0, 1)

    results = {}

    # Тестування
    for model_name, model_path in model_paths.items():
        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, target_names=['Not Dropped Out', 'Dropped Out'],
                                          output_dict=False)
        results[model_name] = report

        print(f"Результати для {model_name}:")
        print(classification_report(y_test, y_pred, target_names=['Not Dropped Out', 'Dropped Out'],
                                          output_dict=False))
        print("=" * 50)

    with open(results_path, 'w') as f:
        for model_name, report in results.items():
            f.write(f"Результати для {model_name}:\n")
            f.write(report)
            f.write("=" * 50 + "\n")

    print("Результати збережено у файл:", results_path)
