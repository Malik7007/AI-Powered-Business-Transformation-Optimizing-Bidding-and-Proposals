from flask import Flask, request, render_template
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Define the base directory as the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load preprocessing objects
with open(os.path.join(base_dir, 'le_dict_go_no_go.pkl'), 'rb') as f:
    le_dict = pickle.load(f)

with open(os.path.join(base_dir, 'scaler_go_no_go.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load trained models
model_paths = {
    "Random Forest": os.path.join(base_dir, 'rf_go_no_go.pkl'),
    "Logistic Regression": os.path.join(base_dir, 'lr_go_no_go.pkl'),
    "SVM": os.path.join(base_dir, 'svc_go_no_go.pkl'),
    "K-Nearest Neighbors": os.path.join(base_dir, 'knn_go_no_go.pkl'),
    "Neural Network": os.path.join(base_dir, 'mlp_go_no_go.pkl'),
    "Naive Bayes": os.path.join(base_dir, 'nb_go_no_go.pkl'),
    "XGBoost": os.path.join(base_dir, 'xgb_go_no_go.pkl'),
}

# Load models and handle potential loading errors
models = {}
for name, path in model_paths.items():
    try:
        models[name] = pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        print(f"Error loading {name} model: {path} not found.")
    except Exception as e:
        print(f"Error loading {name} model: {e}")

# List of categorical columns in the order used for training
categorical_columns = [
    "Project Scope Complexity",
    "Technical Feasibility",
    "Financial Feasibility",
    "Contract Terms",
    "Relevant Experience",
    "Internal Resources Available",
    "Risk Level",
    "Market Demand",
    "Strategic Alignment",
    "Long Term Impact" 
]

# Define the complete feature set (make sure this matches the training set)
all_columns = categorical_columns  # Add any other feature columns here if needed

@app.route('/')
def index():
    return render_template('index.html', categorical_columns=categorical_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve input data from form
    input_data = {}
    for category in categorical_columns:
        input_data[category] = request.form.get(category, "")

    # Convert input data to a DataFrame for processing
    df = pd.DataFrame([input_data], columns=all_columns)

    # Encode categorical features
    for col in all_columns:
        if col in le_dict:
            df[col] = le_dict[col].transform(df[col])
        else:
            return render_template('result.html', predictions={"Error": f"Missing encoder for {col}"})

    # Ensure the DataFrame has the correct column order
    df = df[all_columns]

    # Scale features
    X_scaled = scaler.transform(df)

    # Make predictions with each model and count votes
    predictions = {}
    go_votes = []
    no_go_votes = []
    
    for name, model in models.items():
        if model:  # Ensure the model is loaded
            pred = model.predict(X_scaled)
            pred_label = 'Go' if pred[0] == 0 else 'No-Go'
            predictions[name] = pred_label

            # Append model to the appropriate list based on its prediction
            if pred_label == "Go":
                go_votes.append(name)
            else:
                no_go_votes.append(name)

    # Determine majority vote
    final_decision = "Go" if len(go_votes) > len(no_go_votes) else "No-Go"

    # Pass results to the template
    return render_template('result.html', 
                           predictions=predictions, 
                           final_decision=final_decision, 
                           go_votes=go_votes, 
                           no_go_votes=no_go_votes)

if __name__ == '__main__':
    app.run(debug=True)
