# api.py (Flask backend to serve the signature verification functionality)

from flask import Flask, request, jsonify
import Vinay1  # Import the signature verification script (Vinay1.py)

app = Flask(__name__)

@app.route('/verify-signature', methods=['POST'])
def verify_signature():
    # Extract the signature image path or data (depending on your input method)
    signature_image_path = request.json.get('signature_image_path')

    if not signature_image_path:
        return jsonify({"error": "No signature image path provided"}), 400

    # Call the function from Vinay1.py to verify the signature
    verification_result = Vinay1.signature_verification(signature_image_path)

    return jsonify(verification_result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
