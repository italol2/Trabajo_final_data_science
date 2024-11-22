from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('model.pkl')

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')  # Página principal


# Ruta para predecir en masa desde un archivo CSV
@app.route('/bulk_predict', methods=['POST'])
def bulk_predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    
    # Leer el archivo CSV y preparar los datos
    data = pd.read_csv(file)
    required_columns = ['id', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
                        'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 
                        'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 
                        'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 
                        'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 
                        'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
                        'concavity_worst', 'concave_points_worst', 'symmetry_worst', 
                        'fractal_dimension_worst']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    # Verificar si faltan columnas
    if missing_columns:
        return f"Missing required columns: {', '.join(missing_columns)}", 400
    
    # Predecir
    features = data[required_columns[1:]]  # Excluye 'id' de las características
    predictions = model.predict(features)
    predictions = predictions.astype(int)
    
    # Calcular estadísticas
    total_cases = len(predictions)
    malignant_cases = sum(predictions)
    benign_cases = total_cases - malignant_cases
    malignant_percentage = (malignant_cases / total_cases) * 100
    benign_percentage = (benign_cases / total_cases) * 100
    
    # Filtrar IDs de pacientes con tumores malignos
    data['prediction'] = predictions
    malignant_ids = data[data['prediction'] == 1]['id'].tolist()
    
    # Leer métricas del modelo desde archivo
    df_metrics = pd.read_csv("archivo.csv")
    accuracy = df_metrics['Valores'].iloc[0]
    recall = df_metrics['Valores'].iloc[1]
    f1 = df_metrics['Valores'].iloc[2]
    
    # Renderizar resultados
    return render_template(
        'index.html',
        predictions_summary=True,
        total_cases=total_cases,
        malignant_cases=malignant_cases,
        benign_cases=benign_cases,
        malignant_percentage=round(malignant_percentage, 2),
        benign_percentage=round(benign_percentage, 2),
        accuracy1=round(accuracy, 3) * 100,
        recall1=round(recall, 2) * 100,
        f11=round(f1, 2) * 100,
        malignant_ids=malignant_ids
    )

if __name__ == '__main__':
    app.run(debug=True)