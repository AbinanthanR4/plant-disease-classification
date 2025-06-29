import os
import io
import torch
from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import faiss
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np

from utils.model import ResNet9

app = Flask(__name__)

GEMINI_API_KEY = "your GEMINI API_Key"
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found. Set the GEMINI_API_KEY environment variable.")

PDF_PATH = "plant disease remedies.pdf"
MODEL_PATH = "plant_disease_model.pth"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_DIM = 384

disease_classes = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)__Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape__Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange__Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,bell__Bacterial_spot', 'Pepper,bell__healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Model Loading ---
# Load Plant Disease Model
print("Loading plant disease model...")
try:
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    plant_model = ResNet9(3, len(disease_classes))
    # Load state dict with appropriate map_location
    plant_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True) # Added weights_only for security
    )
    plant_model.to(device) # Move model to the determined device
    plant_model.eval()
    print("Plant disease model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit(1)
except Exception as e:
    print(f"Error loading plant disease model: {e}")
    exit(1)

# Load Sentence Transformer Model
print("Loading sentence embedding model...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("Sentence embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading sentence embedding model: {e}")
    exit(1)

# --- PDF Processing and FAISS Indexing ---
def load_and_process_pdf(pdf_path):
    """Loads PDF, extracts text, and cleans it."""
    print(f"Loading PDF from {pdf_path}...")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            texts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Basic cleaning: replace multiple newlines/spaces
                        cleaned_text = ' '.join(page_text.replace("\n", " ").split())
                        texts.append(cleaned_text)
                    else:
                         print(f"Warning: No text extracted from page {i+1}")
                except Exception as e:
                    print(f"Warning: Could not extract text from page {i+1}: {e}")
            print(f"Extracted text from {len(texts)} pages.")
            return texts
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return None

def create_faiss_index(texts, model):
    """Creates a FAISS index from a list of texts using the provided model."""
    if not texts:
        print("Error: No texts provided to create FAISS index.")
        return None, None
    print("Encoding texts for FAISS index...")
    try:
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        index = faiss.IndexFlatL2(FAISS_INDEX_DIM)
        index.add(embeddings)
        print(f"FAISS index created successfully with {index.ntotal} vectors.")
        return index, embeddings # Returning embeddings might be useful later, but not strictly needed now
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        return None, None

pdf_texts = load_and_process_pdf(PDF_PATH)
if pdf_texts:
    pdf_index, _ = create_faiss_index(pdf_texts, embedding_model)
    if pdf_index is None:
        print("Exiting due to FAISS index creation failure.")
        exit(1)
else:
    print("Exiting due to PDF loading failure.")
    exit(1)


# --- Gemini Configuration ---
print("Configuring Gemini model...")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")
    # Optional: Test connection with a simple prompt
    # gemini_model.generate_content("test")
    print("Gemini model configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini model: {e}")
    # Decide if you want to exit or run without remedy generation
    # For now, let's exit if Gemini setup fails
    exit(1)

# --- Helper Functions ---
def predict_plant_disease(image_bytes):
    """Predicts disease from image bytes."""
    try:
        # Define the same transforms used during training (adjust if needed)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # Add normalization if your model was trained with it
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0) # Add batch dimension

        # Move tensor to the same device as the model
        img_tensor = img_tensor.to(device)

        with torch.no_grad(): # Important for inference
            output = plant_model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)

        predicted_class = disease_classes[predicted_idx.item()]
        confidence_score = confidence.item()

        return predicted_class, confidence_score
    except Exception as e:
        print(f"Error during disease prediction: {e}")
        return None, 0.0

def get_remedy_suggestion(disease_label):
    """Gets remedy suggestion using FAISS retrieval and Gemini generation."""
    if pdf_index is None or not pdf_texts:
        return "Remedy information is currently unavailable."

    try:
        # 1. Retrieve relevant text using FAISS
        query_vec = embedding_model.encode([disease_label], convert_to_numpy=True)
        distances, indices = pdf_index.search(query_vec, k=3) # Retrieve top 3 relevant chunks

        if indices.size == 0 or indices[0][0] == -1: # Check if any valid index found
             print(f"No relevant text found in PDF index for: {disease_label}")
             retrieved_info = "No specific information found in the document."
        else:
             retrieved_texts = [pdf_texts[i] for i in indices[0] if 0 <= i < len(pdf_texts)]
             retrieved_info = "\n\n".join(retrieved_texts)

        # 2. Generate response using Gemini
        prompt = (
            "You are an agricultural assistant specializing in plant diseases. "
            "Based *only* on the information provided below, give concise remedy and prevention steps "
            f"for the plant disease '{disease_label}'. Focus on practical actions a grower can take. "
            "Limit the response to 2-4 sentences or a short bulleted list. If the information is insufficient, state that clearly.\n\n"
            f"--- Relevant Information ---\n{retrieved_info}\n\n"
            f"--- Disease ---\n{disease_label}\n\n"
            "--- Response ---"
        )

        # Use generate_content for potentially simpler API interaction
        response = gemini_model.generate_content(prompt)

        # Accessing response text safely (structure might vary slightly)
        remedy = response.text if hasattr(response, 'text') else getattr(response, 'parts', [{}])[0].get('text', "Could not generate remedy suggestion.")

        return remedy.strip()

    except Exception as e:
        print(f"Error getting remedy suggestion for {disease_label}: {e}")
        return "Could not retrieve or generate remedy suggestion due to an internal error."


# --- API Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    if "plantImage" not in request.files:
        return jsonify({"error": "No file part in the request. Use field name 'plantImage'."}), 400

    file = request.files["plantImage"]

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file:
        try:
            # Read image file bytes
            img_bytes = file.read()

            # Predict disease
            predicted_label, confidence = predict_plant_disease(img_bytes)

            if predicted_label is None:
                 raise ValueError("Disease prediction failed.") # Caught by the outer except

            # Get remedy suggestion
            remedy = get_remedy_suggestion(predicted_label)

            return jsonify({
                "predicted_disease": predicted_label,
                "confidence": f"{confidence:.2f}", # Format confidence
                "remedy_suggestion": remedy
            })

        except Exception as e:
            # Log the detailed error on the server for debugging
            print(f"Error processing request: {e}")
            # Return a generic error to the client
            return jsonify({"error": "Failed to process image.", "details": str(e)}), 500
    else:
        # This case should ideally be caught by the earlier checks, but as a fallback:
         return jsonify({"error": "Invalid file uploaded."}), 400

# --- Main Execution ---
if __name__ == "__main__":
    # Use environment variable for port or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Run the app, accessible on the network
    app.run(host="0.0.0.0", port=port, debug=False) # Turn debug=False for production
 