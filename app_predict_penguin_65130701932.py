import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('penguins_size.csv')

# Data Preprocessing
df = df.dropna(subset=['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])

# Fix SettingWithCopyWarning
df.loc[:, 'sex'] = df['sex'].fillna(df['sex'].mode()[0])
df.loc[:, 'sex'] = df['sex'].replace(to_replace='.', value=df['sex'].mode()[0])

# Encoding categorical variables
species_encoder = LabelEncoder().fit(df['species'])
island_encoder = LabelEncoder().fit(df['island'])
sex_encoder = LabelEncoder().fit(df['sex'])

# Features and target
X = df.drop('species', axis=1)
y = df['species']

# Encode categorical columns using the LabelEncoder
X['sex'] = sex_encoder.transform(X['sex'])
X['island'] = island_encoder.transform(X['island'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline with Random Forest
model = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Normalize the data
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest classifier
])

# Fit the pipeline
model.fit(X_train, y_train)

# Save the model and encoders
with open('model_penguin_65130701932', 'wb') as f:
    pickle.dump((model, species_encoder, island_encoder, sex_encoder), f)

# Function to predict species
def predict_species(sex, island, culmen_length, culmen_depth, flipper_length, body_mass):
    # Prepare the input features
    x_new = pd.DataFrame([[sex_encoder.transform([sex])[0], island_encoder.transform([island])[0],
                           culmen_length, culmen_depth, flipper_length, body_mass]],
                         columns=['sex', 'island', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g'])
    
    # Scale the features
    x_new_scaled = model.named_steps['scaler'].transform(x_new)

    # Make prediction
    y_pred_new = model.predict(x_new_scaled)
    result = species_encoder.inverse_transform(y_pred_new)
    
    return result[0]

# Streamlit UI
st.title('Penguin Species Prediction')
st.write('Enter the penguin details below to predict the species:')

# Input fields
sex = st.selectbox('Sex:', ['Male', 'Female'])
island = st.selectbox('Island:', df['island'].unique())
culmen_length = st.number_input('Culmen Length (mm):', min_value=0.0, step=0.1)
culmen_depth = st.number_input('Culmen Depth (mm):', min_value=0.0, step=0.1)
flipper_length = st.number_input('Flipper Length (mm):', min_value=0.0, step=0.1)
# Fixing the StreamlitMixedNumericTypesError by using consistent types for `value` and `step`
body_mass = st.number_input('Body Mass (g):', min_value=0.0, step=0.1, value=0.0)


# Predict button
if st.button('Predict'):
    prediction = predict_species(sex, island, culmen_length, culmen_depth, flipper_length, body_mass)
    st.write(f'Predicted Species: {prediction}')

    # Model Evaluation (Accuracy & Confusion Matrix)
    accuracy = model.score(X_test, y_test)
    st.write(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=species_encoder.classes_,
                yticklabels=species_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)



