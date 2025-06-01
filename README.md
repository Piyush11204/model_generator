Model Generator
A Flask-based web application to upload CSV files, generate machine learning models, store metadata in MongoDB, create usage documentation, and allow liking and downloading models.
Setup Instructions

Prerequisites:

Python 3.9+
MongoDB running locally or via MongoDB Atlas
pdflatex installed for PDF generation (e.g., TeX Live)


Installation:
git clone <repository_url>
cd model_generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Environment Variables:Create a .env file in the project root:
SECRET_KEY=your-secret-key
MONGODB_URI=mongodb://localhost:27017


Run the Application:
python app.py

Access the app at http://localhost:5000.

Run Tests:
pytest tests/test_app.py



Usage

Upload a Model:

Go to the homepage and fill out the form with a model name, description, and CSV file.
The CSV should have features and a target column (last column by default).
The app generates a Random Forest model and a PDF usage document.


View Models:

Models are displayed in a grid with names, descriptions, like counts, and download links.
Click "Like" to increment the like count (one like per IP).


Download:

Download the .pkl model file or the PDF documentation for any model.



Dependencies
See requirements.txt for a full list of dependencies.
Notes

Ensure MongoDB is running before starting the app.
CSV files must be under 10MB and have valid data (numerical/categorical features, target column).
The app uses Tailwind CSS via CDN for styling.
PDF generation requires pdflatex installed.

Project Structure
model_generator/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── models/                   # Store .pkl model files
├── docs/                     # Store generated PDF documents
├── uploads/                  # Store uploaded CSV files
├── templates/                # HTML templates
├── static/js/                # JavaScript for client-side logic
├── utils/                    # Utility scripts for processing, PDF, and DB
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
├── .env                      # Environment variables
└── README.md                 # This file

