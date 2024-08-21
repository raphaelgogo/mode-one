from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the saved TensorFlow model
model = tf.keras.models.load_model('malari_model.keras')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        # Extracting the data from the request
        wbc_count = request.form.get('wbc_count', 0)
        rcb_count = request.form.get('rcb_count', 0)
        hb_level = request.form.get('hb_level', 0)
        hematocrit = request.form.get('hematocrit', 0)
        mean_cell_volume = request.form.get('mean_cell_volume', 0)
        mean_corp_hb = request.form.get('mean_corp_hb', 0)
        mean_cell_hb_conc = request.form.get('mean_cell_hb_conc', 0)
        platelet_count = request.form.get('platelet_count', 0)
        platelet_distr_width = request.form.get('platelet_distr_width', 0)
        mean_platelet_vl = request.form.get('mean_platelet_vl', 0)
        neutrophils_percent = request.form.get('neutrophils_percent', 0)
        lymphocytes_percent = request.form.get('lymphocytes_percent', 0)
        mixed_cell_percent = request.form.get('mixed_cell_percent', 0)
        neutrophils_count = request.form.get('neutrophils_count', 0)
        lymphocytes_count = request.form.get('lymphocytes_count', 0)
        mixed_cell_count = request.form.get('mixed_cell_count', 0)
        rcb_distr_width_percent = request.form.get('rcb_distr_width_percent', 0)

      

        # Check if any of the inputs are missing or None
        input_list = [wbc_count, rcb_count, hb_level, hematocrit, mean_cell_volume, 
                      mean_corp_hb, mean_cell_hb_conc, platelet_count, platelet_distr_width, 
                      mean_platelet_vl, neutrophils_percent, lymphocytes_percent, mixed_cell_percent, 
                      neutrophils_count, lymphocytes_count, mixed_cell_count, rcb_distr_width_percent]
        
        if None in input_list:
            missing_fields = [field for field, value in zip(
                ['wbc_count', 'rcb_count', 'hb_level', 'hematocrit', 'mean_cell_volume', 
                 'mean_corp_hb', 'mean_cell_hb_conc', 'platelet_count', 'platelet_distr_width', 
                 'mean_platelet_vl', 'neutrophils_percent', 'lymphocytes_percent', 'mixed_cell_percent', 
                 'neutrophils_count', 'lymphocytes_count', 'mixed_cell_count', 'rcb_distr_width_percent'], 
                input_list) if value is None]
            return jsonify({'error': f'Missing input fields: {", ".join(missing_fields)}'}), 400
        
        # Convert inputs to float
        input_query = np.array([[float(wbc_count), float(rcb_count), float(hb_level), float(hematocrit), 
                                 float(mean_cell_volume), float(mean_corp_hb), float(mean_cell_hb_conc), 
                                 float(platelet_count), float(platelet_distr_width), float(mean_platelet_vl), float(neutrophils_percent),
                                 float(lymphocytes_percent), float(mixed_cell_percent), float(neutrophils_count), 
                                 float(lymphocytes_count), float(mixed_cell_count), float(rcb_distr_width_percent)]])
        
        #Optional: Apply the same scaling if the model was trained on scaled data
        #scaler = MinMaxScaler()
        #input_query = scaler.transform(input_query)  # Uncomment and ensure the scaler is fitted with training data

        # Make prediction
        result = model.predict(input_query)[0]

        # Format the prediction result
        return jsonify({'predicted_value': str(result)})

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
