import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('trained_model.pkl')
    return model

# Feature importance function
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    st.pyplot(plt)

# Main function for the app
def main():
    model = load_model()

    st.title("Insurance Charges Prediction App")
    st.write("Predict insurance charges based on personal attributes and explore additional features.")

    # Sidebar for user options
    option = st.sidebar.selectbox(
        "Choose an option:",
        ("Single Prediction", "Batch Prediction", "View Feature Importance", "Upload Dataset")
    )

    if option == "Single Prediction":
        st.subheader("Single Prediction")

        # Input fields for single prediction
        age = st.number_input("Age", value=30, step=1, min_value=0, max_value=120)
        sex = st.selectbox("Sex", options=["Male", "Female"])
        bmi = st.number_input("BMI (Body Mass Index)", value=25.0, step=0.1, min_value=0.0, max_value=60.0)
        children = st.number_input("Number of Children", value=0, step=1, min_value=0)
        smoker = st.selectbox("Smoker", options=["Yes", "No"])
        region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

        # Map categorical inputs
        sex_numeric = 1 if sex == "Male" else 0
        smoker_numeric = 1 if smoker == "Yes" else 0
        region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
        region_numeric = region_mapping[region]

        features = np.array([age, sex_numeric, bmi, children, smoker_numeric, region_numeric]).reshape(1, -1)

        if st.button("Predict Charges"):
            predicted_charges = model.predict(features)[0]
            st.success(f"Predicted Insurance Charges: ${predicted_charges:,.2f}")

    elif option == "Batch Prediction":
        st.subheader("Batch Prediction")

        uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            # Ensure the required columns are present
            required_columns = ["age", "sex", "bmi", "children", "smoker", "region"]
            if all(col in data.columns for col in required_columns):
                # Map categorical columns
                data["sex"] = data["sex"].map({"Male": 1, "Female": 0})
                data["smoker"] = data["smoker"].map({"Yes": 1, "No": 0})
                region_mapping = {"Northeast": 0, "Northwest": 1, "Southeast": 2, "Southwest": 3}
                data["region"] = data["region"].map(region_mapping)

                # Predict charges
                predictions = model.predict(data[required_columns])
                data["predicted_charges"] = predictions
                st.write("Batch Predictions:")
                st.dataframe(data)
                st.download_button(
                    "Download Predictions",
                    data.to_csv(index=False),
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error("The uploaded CSV does not contain the required columns.")

    elif option == "View Feature Importance":
        st.subheader("Feature Importance")
        feature_names = ["age", "sex", "bmi", "children", "smoker", "region"]
        plot_feature_importance(model, feature_names)

    elif option == "Upload Dataset":
        st.subheader("Upload Dataset")

        uploaded_file = st.file_uploader("Upload a dataset to preview", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(data)

            st.write("Dataset Summary:")
            st.write(data.describe(include="all"))

# Run the app
if __name__ == "__main__":
    main()
