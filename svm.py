import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

class SVMDetector:
    def __init__(self, file_path, features_to_exclude):
        self.data = pd.read_csv(file_path)
        self.features_to_exclude = features_to_exclude
        self.X = None
        self.y = None
        self.model = SVC(kernel='linear')  # Consider changing the kernel based on your dataset

    def preprocess_data(self):
        self.data = self.data.drop(columns=self.features_to_exclude)
        self.X = self.data.drop('Class', axis=1)  # Assuming 'Class' is your target variable
        self.y = self.data['Class']
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        print(classification_report(y_test, predictions))
        print("Accuracy:", accuracy_score(y_test, predictions))

if __name__ == "__main__":
    file_path = 'creditcard.csv'  # Update this with your actual file path
    features_to_exclude = ['feature1', 'feature2', ...]  # Fill in with features to exclude
    svm_detector = SVMDetector(file_path, features_to_exclude)
    svm_detector.preprocess_data()
    svm_detector.train_model()
