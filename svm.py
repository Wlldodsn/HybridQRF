import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

class SVMDetector:
    def __init__(self, file_path, features_to_use):
        self.data = pd.read_csv(file_path)
        self.features_to_use = features_to_use  # List of features to use
        self.X = None
        self.y = None

        # Linear version for kernel
        # self.model = SVC(kernel='linear', probability=True)  # probability parameter set to True

        # Non-linear version for kernel
        self.model = SVC(probability=True)  # Removed kernel='linear'

    def preprocess_data(self):
        self.data = self.data[self.features_to_use + ['Class']]  # Keep only the specified features and the target
        self.X = self.data.drop('Class', axis=1)
        self.y = self.data['Class']
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    # Linear version for train_model
    #def train_model(self):
    #    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    #    self.model.fit(X_train, y_train)
    #    predictions = self.model.predict(X_test)
    #    # Get the probabilities for the positive class (assuming it's the second class)
    #    probabilities = self.model.predict_proba(X_test)[:, 1]
    #    auc = roc_auc_score(y_test, probabilities)
    #    print(classification_report(y_test, predictions))
    #    print("Accuracy:", accuracy_score(y_test, predictions))
    #    print("AUC:", auc)  # Print AUC score
        
    # 
    #def train_model(self):
    #    X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
    #    # Define the parameter grid to search
    #    param_grid = {
    #        'C': [0.1, 1, 10],
    #        'gamma': [0.001, 0.0001, 'scale'],  # 'scale' uses 1 / (n_features * X.var()) as value of gamma
    #        'kernel': ['rbf']  # You can add more kernels to try
    #    }
    #    grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='roc_auc', verbose=2)
    #    grid_search.fit(X_train, y_train)
    #
    #    print("Best Parameters:", grid_search.best_params_)
    #    best_model = grid_search.best_estimator_
    #    predictions = best_model.predict(X_test)
    #    probabilities = best_model.predict_proba(X_test)[:, 1]
    #    auc = roc_auc_score(y_test, probabilities)
    #    
    #    # Output results
    #    print(classification_report(y_test, predictions))
    #    print("Accuracy:", accuracy_score(y_test, predictions))
    #    print("AUC:", auc)
        
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42) # consider changing test size
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.0001, 'scale'], 
            'kernel': ['rbf', 'linear', 'poly']  # Trying different kernels
        }
        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring='roc_auc', verbose=2)
        grid_search.fit(X_train, y_train)

        # After fitting, print the best parameters
        print("Best Parameters:", grid_search.best_params_)

        # To see the performance of each combination
        print("\nAll Results:")
        results = grid_search.cv_results_
        for mean_score, params in zip(results['mean_test_score'], results['params']):
            print(params, '-> ROC AUC:', mean_score)

        # Proceed to evaluate the best model found by GridSearchCV
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        probabilities = best_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        
        # Output the evaluation metrics for the best model
        print(classification_report(y_test, predictions))
        print("Accuracy:", accuracy_score(y_test, predictions))
        print("AUC:", auc)

if __name__ == "__main__":
    file_path = 'creditcard.csv'  # Update with your actual file path
    # Based on your Random Forest feature importance, use up to the top 21 features
    features_to_use = ['V17', 'V12', 'V14', 'V10', 'V16', 'V11', 'V9', 'V7', 'V18', 'V4', 'V21', 'V3', 'V26', 'V20', 'V2', 'V1', 'V19', 'V6', 'Amount', 'V8', 'V15']

    # Without random forest, uses all 29 features
    # features_to_use = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    
    svm_detector = SVMDetector(file_path, features_to_use)
    svm_detector.preprocess_data()
    svm_detector.train_model()