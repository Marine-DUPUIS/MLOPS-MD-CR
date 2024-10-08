import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np

# Chargement du modèle
model_uri = 'runs:/a93938c195e648719c4e3a42a21565d7/modele_regression_logistique'
model = mlflow.sklearn.load_model(model_uri)

st.title("Application - prédiction du risque de défaut de crédit")

# Création de champs de saisie 
credit_lines = st.number_input('Nombre de lignes de crédit', min_value=0, max_value=100)
loan_amt = st.number_input('Montant du prêt restant', min_value=0.0)
total_debt = st.number_input('Dette totale restante', min_value=0.0)
income = st.number_input('Revenu', min_value=0.0)
years_employed = st.number_input('Années d\'emploi', min_value=0, max_value=100)
fico_score = st.number_input('Score FICO', min_value=300, max_value=850)

# Conversion des entrées de l'utilisateur en un tableau pour la prédiction
input_data = np.array([[credit_lines, loan_amt, total_debt, income, years_employed, fico_score]])

if st.button('Prédire'):
    # Affichage de la prédiction avec le modèle
    prediction = model.predict(input_data)
    if prediction == 1:
        st.error('Risque élevé de défaut de paiement.')
    else:
        st.success('Faible risque de défaut de paiement.')
