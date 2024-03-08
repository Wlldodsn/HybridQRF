# https://www.researchgate.net/publication/374083997_FRAUD_DETECTION_USING_MACHINE_LEARNING

import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

class FraudDetection:
    def __init__(self):
        self.data = None
        self.model = None
        self.X = None
        self.y = None

    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)

    def preprocess_data(self):
        # Data Encoding
        label_encoder = LabelEncoder()
        self.data['region'] = label_encoder.fit_transform(self.data['region'])
        self.data['state'] = label_encoder.fit_transform(self.data['state'])
        self.data['customer_category'] = label_encoder.fit_transform(self.data['customer_category'])

        # Perform one-hot encoding on categorical variables
        categorical_cols = ['region', 'state', 'customer_categoroy']
        self.data_encoded = pd.get_dummies(self.data, columns = categorical_cols)

        # Preprocessing steps
        self.data_encoded.drop(['step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)
        
        # Split the dataset into features (X) and labels (y)
        self.X = self.data_encoded.drop('isFraud', axis=1)
        self.y = self.data_encoded['isFraud']

        # Scale the numerical features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def train_model(self, model_type='RandomForest'):
        if model_type == 'RandomForest':
            self.model = RandomForestClassifier()
        elif model_type == 'SVM': 
            self.model = SVC()
        elif model_type == 'KMeans':
            self.model = KMeans(n_clusters=2, random_state=42)
        else:
            raise ValueError("invalid model type. Please choose 'RandomForest', 'SVM', or 'KMeans'.")
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X_scaled, self.y, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))

    def load_model(self, model_file):
        self.model = joblib.load(model_file)

    def make_prediction(self, sample_data):
        # Perform the necessary preprocessing on the sample_data DataFrame to match the training data format

        # Use the loaded model to make predictions on the input data
        predictions = self.model.predict(sample_data)

        return predictions
    
    if __name__ == "__main__":
        # Instantiate the FraudDetection class
        fraud_detector = FraudDetection()

        # Load the dataset
        fraud_detector.load_data('financial_transactions.csv')

        # Preprocess the data
        fraud_detector.preprocess_data()

        # Train and evaluate the model
        fraud_detector.train_model(model_type='RandomForest')

        # Load the trained model
        fraud_detector.load_model('model.pkl')

        # Sample data for prediction (replace this with your sample data)
        sample_data = pd.DataFrame(...) # Provide your sample data as a DataFrame

        # Make a sample prediction
        predictions = fraud_detector.make_prediction(sample_data)
        print(predictions)