require('dotenv').config();

const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data'); // Explicitly require form-data
const fs = require('fs');
const path = require('path');
const cors = require('cors');

const app = express();

// --- Configuration ---
const EXPRESS_PORT = process.env.PORT || 3000; // Port for this Express server
const FLASK_API_URL = process.env.FLASK_API_URL || 'http://127.0.0.1:5000/predict'; // URL of your Python Flask API
const UPLOAD_DIR = path.join(__dirname, 'uploads'); // Directory for temporary uploads
const MAX_FILE_SIZE = 16 * 1024 * 1024; // 16 MB limit

// --- Initial Setup ---
// Ensure the upload directory exists
try {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    console.log(`Upload directory ensured at: ${UPLOAD_DIR}`);
} catch (error) {
    console.error(`Error creating upload directory: ${UPLOAD_DIR}`, error);
    process.exit(1); // Exit if we can't create the upload dir
}

// --- Middleware ---
app.use(cors()); // Enable Cross-Origin Resource Sharing

// Basic request logger
app.use((req, res, next) => {
    console.log(`[Express] ${new Date().toISOString()} - ${req.method} ${req.originalUrl}`);
    next();
});

// --- Multer Configuration (for handling file uploads) ---
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, UPLOAD_DIR); // Save files to the 'uploads' directory
    },
    filename: function (req, file, cb) {
        // Generate a unique filename to avoid collisions
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        const extension = path.extname(file.originalname);
        cb(null, file.fieldname + '-' + uniqueSuffix + extension);
    }
});

const upload = multer({
    storage: storage,
    limits: { fileSize: MAX_FILE_SIZE }, // Apply file size limit
    fileFilter: function (req, file, cb) {
        // Accept only common image file types
        const allowedExtensions = /\.(jpg|jpeg|png|gif|bmp|webp)$/i;
        const isMimeTypeAllowed = file.mimetype.startsWith('image/'); // Check MIME type as well

        if (allowedExtensions.test(file.originalname) && isMimeTypeAllowed) {
            console.log(`[Express Multer] Accepting file: ${file.originalname}`);
            cb(null, true); // Accept the file
        } else {
            console.warn(`[Express Multer] REJECTING file: ${file.originalname} (MIME: ${file.mimetype})`);
            // Create a specific error for invalid file type
            const error = new Error('Invalid file type. Only image files (jpg, jpeg, png, gif, bmp, webp) are allowed.');
            error.code = 'INVALID_FILE_TYPE'; // Custom error code
            cb(error, false); // Reject the file
        }
    }
    // The field name in the form sending the file must be 'plantImage'
}).single('plantImage');

// --- API Routes ---

// Route to handle image prediction requests
app.post('/predict', (req, res) => {
    upload(req, res, async (uploadError) => {
        // --- Handle Multer Errors ---
        if (uploadError) {
            console.error("[Express] Multer Upload Error:", uploadError.message);
            if (uploadError.code === 'LIMIT_FILE_SIZE') {
                return res.status(400).json({
                    error: "File Too Large",
                    details: `Maximum file size allowed is ${MAX_FILE_SIZE / 1024 / 1024} MB.`
                });
            }
             if (uploadError.code === 'INVALID_FILE_TYPE') {
                return res.status(400).json({
                    error: "Invalid File Type",
                    details: uploadError.message
                });
            }
            // Generic upload error
            return res.status(400).json({
                error: "File Upload Error",
                details: uploadError.message || "Could not process file upload."
            });
        }

        // --- Handle Case Where No File Was Uploaded ---
        if (!req.file) {
            console.error("[Express] No file uploaded or processed by Multer.");
            return res.status(400).json({
                error: "Bad Request",
                details: "No valid image file found in the 'plantImage' field."
            });
        }

        // --- File Successfully Received ---
        const tempFilePath = req.file.path;
        const originalFilename = req.file.originalname;
        console.log(`[Express] File received: '${originalFilename}', Temp path: ${tempFilePath}`);

        // --- Prepare FormData for Flask ---
        const formDataToFlask = new FormData();
        let fileStream;
        try {
            // Create a readable stream from the temporary file
            fileStream = fs.createReadStream(tempFilePath);
            // Append the stream to the FormData object
            // IMPORTANT: Flask expects the field name 'plantImage'
            formDataToFlask.append('plantImage', fileStream, { filename: originalFilename });
        } catch (streamError) {
            console.error(`[Express] Error creating read stream: ${streamError.message}`);
            // Clean up the temp file if stream creation fails
            fs.unlink(tempFilePath, (unlinkErr) => {
                if (unlinkErr) console.error(`[Express] Error deleting temp file after stream error: ${tempFilePath}`, unlinkErr);
            });
            return res.status(500).json({ error: "Internal Server Error", details: "Failed to read uploaded file." });
        }

        // --- Call Flask API ---
        try {
            console.log(`[Express] Forwarding request to Flask: ${FLASK_API_URL}`);
            const flaskResponse = await axios.post(FLASK_API_URL, formDataToFlask, {
                headers: {
                    // Crucial: Let axios set the Content-Type header using form-data library's calculations
                    ...formDataToFlask.getHeaders()
                },
                timeout: 120000 // 2 minute timeout for potentially long processing
            });

            console.log(`[Express] Flask response received (Status: ${flaskResponse.status})`);
            // Send the Flask response back to the original client
            res.status(flaskResponse.status).json(flaskResponse.data);

        } catch (axiosError) {
            console.error(`[Express] Error calling Flask API: ${axiosError.message}`);

            // Handle different types of axios errors
            if (axiosError.response) {
                // The request was made and the server responded with a status code
                // that falls out of the range of 2xx
                console.error("[Express] Flask Error Response Data:", axiosError.response.data);
                console.error("[Express] Flask Error Response Status:", axiosError.response.status);
                // Forward the error status and data from Flask
                res.status(axiosError.response.status).json(
                    axiosError.response.data || { error: "Backend Error", details: "Flask API returned an error." }
                );
            } else if (axiosError.request) {
                // The request was made but no response was received
                console.error("[Express] No response received from Flask API.");
                res.status(504).json({ error: "Gateway Timeout", details: "Did not receive a response from the backend service." });
            } else {
                // Something happened in setting up the request that triggered an Error
                console.error('[Express] Error setting up Flask request:', axiosError.message);
                res.status(500).json({ error: "Internal Server Error", details: "Failed to configure request to backend service." });
            }

        } finally {
            // --- Clean up the temporary file ---
            // Use async unlink and handle potential errors
            fs.unlink(tempFilePath, (unlinkErr) => {
                if (unlinkErr) {
                    console.error(`[Express] Failed to delete temp file: ${tempFilePath}`, unlinkErr);
                } else {
                    console.log(`[Express] Temp file deleted: ${tempFilePath}`);
                }
            });
        }
    });
});

// --- Root Route ---
app.get('/', (req, res) => {
    res.status(200).send('✅ Express Proxy for Flask Plant Disease API is running. POST an image file to /predict with field name "plantImage".');
});

// --- Global Error Handler (for errors not caught in routes) ---
app.use((err, req, res, next) => {
    console.error("[Express Global Error Handler]", err.stack || err);
    // Avoid sending detailed stack traces to the client in production
    res.status(500).json({
        error: 'Internal Server Error',
        details: process.env.NODE_ENV === 'production' ? 'An unexpected error occurred.' : err.message
     });
});

// --- Start Server ---
app.listen(EXPRESS_PORT, () => {
    console.log(`\n🚀 Express proxy server running on http://localhost:${EXPRESS_PORT}`);
    console.log(`   Temporarily storing uploads in: ${UPLOAD_DIR}`);
    console.log(`   Forwarding '/predict' requests to Flask at: ${FLASK_API_URL}`);
    console.log(`   Max file size: ${MAX_FILE_SIZE / 1024 / 1024} MB\n`);
});