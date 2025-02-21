// server.js (Node.js backend to serve the interface and interact with Flask API)

const express = require('express');
const path = require('path');
const FormData = require('form-data');
const fetch = require('node-fetch');  // For sending HTTP requests to Flask API

const app = express();
const port = 3000;

// Serve static files (e.g., HTML, JS, CSS)
app.use(express.static(path.join(__dirname, 'public')));

// Middleware to handle file uploads
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Endpoint to handle the signature upload and forward the request to Flask
app.post('/verify-signature', async (req, res) => {
    const file = req.files.signature; // Assuming the form uses 'signature' as the field name

    // Prepare data for Flask API (you can adjust this if Flask expects data differently)
    const form = new FormData();
    form.append("signature_image_path", file.path);

    try {
        const response = await fetch('http://localhost:5000/verify-signature', {
            method: 'POST',
            body: form
        });

        const data = await response.json();
        res.json(data); // Return the verification result back to the frontend
    } catch (error) {
        console.error('Error communicating with Flask API:', error);
        res.status(500).send('Error verifying signature.');
    }
});

// Start the server
app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});

