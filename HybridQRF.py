from RandomForest import FraudDetection

def run_qsvc(data):
    #TODO: implement
    pass

def run_rf(filepath, threshold, n_features_to_select):

    
    # Create an instance of the class
    fraud_detector = FraudDetection(csv_file_path)

    # Preprocess data
    fraud_detector.preprocess_data()
    
    # Train the model and print out classification report
    auc_score = fraud_detector.train_model()
    print(f"Initial AUC Score: {auc_score}")

    # Evaluate the importance of features based on cross-validated performance
    trimmed_data, feature_names, num_features = fraud_detector.evaluate_feature_importance(n_features_to_select=n_features_to_select, threshold=threshold)  
    
    print(f"{num_features} important features were found: {feature_names}")


    # Save the model
    fraud_detector.save_model('rf_model.pkl')

    return fraud_detector, trimmed_data



if __name__ == "__main__":
    # Path to your CSV file
    csv_file_path = 'creditcard.csv'

    threshold = 0.99    #tune
    n_features_to_select = 29   #tune

    fraud_detector, trimmed_data = run_rf(csv_file_path, threshold, n_features_to_select)

    #outputs = run_qsvc(trimmed_data)