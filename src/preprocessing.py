import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib


def load_and_preprocess_data(file_path):
    """
    Load and preprocess data.

    Args:
        file_path (str): Path to the CSV file.
        target_column (str): The name of the target column.
        categorical_columns (list): List of categorical columns to encode.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for train-test split.

    Returns:
        tuple: (x_train, x_test, y_train, y_test, feature_names)
    """
    # Load dataset
    df = pd.read_csv(file_path)


    # Encode categorical columns
    label_encoders = {}
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        label_encoders[col] = encoder

    # Split into features and target
    X = df.drop('charges', axis=1)
    y = df['charges']

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.2
    )

    return x_train, x_test, y_train, y_test


def save_preprocessing_artifacts(feature_names, feature_names_path, label_encoders, encoders_path):
    """
    Save preprocessing artifacts to files.

    Args:
        feature_names (list): List of feature names.
        feature_names_path (str): Path to save feature names.
        label_encoders (dict): Dictionary of LabelEncoders for categorical features.
        encoders_path (str): Path to save LabelEncoders.
    """
    # Save feature names
    with open(feature_names_path, 'w') as f:
        f.write('\n'.join(feature_names))

