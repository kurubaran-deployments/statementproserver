
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import time
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load the trained model and vectorizer
model = joblib.load('sub_category_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/upload', methods=['POST'])
def upload_file():
    user_email = request.headers.get('User-Email')
    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.xlsx'):
        # Create directories if they don't exist
        user_upload_dir = os.path.join('Uploads', user_email, 'RawData')
        user_output_dir = os.path.join('Output', user_email)
        os.makedirs(user_upload_dir, exist_ok=True)
        os.makedirs(user_output_dir, exist_ok=True)

        upload_path = os.path.join(user_upload_dir, file.filename)
        if os.path.exists(upload_path):
            return jsonify({"error": "File already exists"}), 400

        # Save the uploaded file
        file.save(upload_path)

        # Process the file
        raw_data = pd.read_excel(upload_path, skiprows=6)

        # Predict sub categories
        descriptions = raw_data['Description'].astype(str)
        descriptions_vectorized = vectorizer.transform(descriptions)
        raw_data['Sub Category'] = model.predict(descriptions_vectorized)

        raw_data['Transaction Date'] = raw_data['Date']
        raw_data['Adjusted Amount'] = raw_data['Amount']
        raw_data['Actual Amount'] = raw_data['Amount']
        raw_data['Income Amount'] = raw_data['Amount'].apply(lambda x: x if x > 0 else 0)
        raw_data['Expense Amount'] = raw_data['Amount'].apply(lambda x: -x if x < 0 else 0)
        raw_data['Comments'] = ''
        raw_data['Additional Notes'] = ''

        final_data = raw_data[
            ['Transaction Date', 'Description', 'Sub Category', 'Adjusted Amount', 'Actual Amount', 'Income Amount',
             'Expense Amount', 'Comments', 'Additional Notes']]

        # Save the final CSV file with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_filename = f'final_{timestamp}.csv'
        output_path = os.path.join(user_output_dir, output_filename)
        final_data.to_csv(output_path, index=False)

        return jsonify({"message": "File processed successfully", "output_file": output_filename}), 200

    return jsonify({"error": "Unsupported file type"}), 400

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    user_email = request.headers.get('User-Email')
    if not user_email:
        return jsonify({"error": "User email is required"}), 400

    user_output_dir = os.path.join('Output', user_email)
    return send_from_directory(user_output_dir, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=6010)
