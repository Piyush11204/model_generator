import os
import uuid
import time
import logging
import threading
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for, send_file, abort, jsonify
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from pathlib import Path
from queue import Queue

# Import project configuration
from config import Config

# Import our enhanced ML trainer
from utils.ml_trainer import MLModelTrainer
from utils.pdf_generator import generate_enhanced_pdf
from utils.db import save_model_metadata, get_all_models, save_like, has_user_liked, get_models_from_filesystem

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)  # Initialize directories

# Ensure directories exist
for folder in [app.config['UPLOAD_FOLDER'], app.config['DOC_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Training job queue
training_queue = Queue()
training_status = {}

# Try to connect to MongoDB
try:
    client = MongoClient(app.config['MONGODB_URI'])
    db = client[app.config['MONGODB_DB']]
    db.models.create_index('model_id')
    db.likes.create_index([('model_id', 1), ('user_ip', 1)])
    db.training_jobs.create_index('job_id')
    logger.info("Connected to MongoDB")
except ConnectionFailure as e:
    logger.error(f"MongoDB connection failed: {e}")
    db = None  # Will fallback to filesystem storage

# Available model configurations
MODEL_CONFIGS = {
    'fast': {
        'models': ['random_forest', 'logistic_regression', 'naive_bayes'],
        'hyperparameter_tuning': False,
        'description': 'Quick training with basic models'
    },
    'balanced': {
        'models': ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm'],
        'hyperparameter_tuning': True,
        'description': 'Balanced performance and training time'
    },
    'comprehensive': {
        'models': None,  # All models
        'hyperparameter_tuning': True,
        'description': 'Train all available models with hyperparameter tuning'
    },
    'ensemble': {
        'models': ['random_forest', 'gradient_boosting', 'extra_trees'],
        'hyperparameter_tuning': True,
        'description': 'Focus on ensemble methods for best performance'
    }
}

PREPROCESSING_CONFIGS = {
    'standard': {
        'imputation_strategy': 'simple_median',
        'scaling_method': 'standard'
    },
    'robust': {
        'imputation_strategy': 'knn',
        'scaling_method': 'robust'
    },
    'minimal': {
        'imputation_strategy': 'simple_mean',
        'scaling_method': 'minmax'
    }
}

def background_training_worker():
    """Background worker to process training jobs."""
    while True:
        try:
            if not training_queue.empty():
                job = training_queue.get()
                process_training_job(job)
                training_queue.task_done()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Training worker error: {e}")

def process_training_job(job):
    """Process a single training job."""
    job_id = job['job_id']
    try:
        training_status[job_id]['status'] = 'training'
        training_status[job_id]['progress'] = 20
        
        # Initialize ML trainer
        trainer = MLModelTrainer(base_dir=app.config['BASE_DIR'])
        
        # Update progress
        training_status[job_id]['progress'] = 40
        training_status[job_id]['message'] = 'Preprocessing data...'
        
        # Run training pipeline
        results = trainer.run_complete_pipeline(
            csv_path=job['csv_path'],
            models_to_train=job['models_to_train'],
            imputation_strategy=job['imputation_strategy'],
            scaling_method=job['scaling_method'],
            test_size=job['test_size'],
            hyperparameter_tuning=job['hyperparameter_tuning'],
            experiment_name=job['experiment_name']
        )
        
        # Update progress
        training_status[job_id]['progress'] = 80
        training_status[job_id]['message'] = 'Generating documentation...'
        
        # Generate enhanced PDF
        doc_path = os.path.join(app.config['DOC_FOLDER'], f"{job['model_id']}.pdf")
        generate_enhanced_pdf(
            model_name=job['model_name'],
            description=job['description'],
            results=results,
            model_id=job['model_id'],
            doc_path=doc_path
        )
        
        # Save enhanced metadata to MongoDB
        metadata = {
            'model_id': job['model_id'],
            'name': job['model_name'],
            'description': job['description'],
            'upload_date': datetime.utcnow(),
            'like_count': 0,
            'model_files': results['saved_files'],
            'document_file_path': doc_path,
            'training_config': {
                'models_trained': list(results['model_results'].keys()),
                'best_model': results['model_results']['best_model'],
                'best_accuracy': results['model_results']['best_accuracy'],
                'preprocessing_config': results['preprocessing_config']
            },
            'data_summary': results['data_summary']
        }
        
        if db is not None:  # Changed from 'if db:' to 'if db is not None:'
            save_model_metadata(db, metadata)
        
        # Update training status
        training_status[job_id]['status'] = 'completed'
        training_status[job_id]['progress'] = 100
        training_status[job_id]['message'] = 'Training completed successfully!'
        training_status[job_id]['results'] = {
            'best_model': results['model_results']['best_model'],
            'best_accuracy': results['model_results']['best_accuracy'],
            'models_trained': len(results['model_results']) - 2  # Exclude best_model and best_accuracy keys
        }
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {e}")
        training_status[job_id]['status'] = 'failed'
        training_status[job_id]['message'] = f'Training failed: {str(e)}'

# Start background worker
training_thread = threading.Thread(target=background_training_worker, daemon=True)
training_thread.start()

@app.route('/')
def index():
    """Render homepage with upload form and model list."""
    # Get models from database if available
    models = []
    
    if db is not None:  # Changed from 'if db:' to 'if db is not None:'
        models = get_all_models(db)
    
    # Get models from file system as fallback
    if not models:
        models_dir = app.config['MODEL_FOLDER']
        models = get_models_from_filesystem(
            str(models_dir), 
            app.config['DOC_FOLDER']
        )
    
    # Sort by upload date (newest first)
    models.sort(key=lambda x: x.get('upload_date', datetime.min), reverse=True)
    
    return render_template('index.html', 
                         models=models,
                         model_configs=MODEL_CONFIGS,
                         preprocessing_configs=PREPROCESSING_CONFIGS)

@app.route('/upload', methods=['POST'])
def upload():
    """Handle CSV upload and initiate model training."""
    if 'file' not in request.files:
        flash('No file uploaded', 'error')
        return redirect(url_for('index'))
    
    file = request.files['file']
    model_name = request.form.get('model_name')
    description = request.form.get('description')
    training_mode = request.form.get('training_mode', 'balanced')
    preprocessing_mode = request.form.get('preprocessing_mode', 'standard')
    test_size = float(request.form.get('test_size', 0.2))
    
    if not file or not model_name or not description:
        flash('All fields are required', 'error')
        return redirect(url_for('index'))
    
    if not file.filename.endswith('.csv'):
        flash('Only CSV files are allowed', 'error')
        return redirect(url_for('index'))
    
    try:
        # Generate unique identifiers
        model_id = str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        
        # Save uploaded file
        filename = secure_filename(f"{model_id}.csv")
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(csv_path)
        
        # Get configuration
        model_config = MODEL_CONFIGS[training_mode]
        preprocessing_config = PREPROCESSING_CONFIGS[preprocessing_mode]
        
        # Create training job
        training_job = {
            'job_id': job_id,
            'model_id': model_id,
            'model_name': model_name,
            'description': description,
            'csv_path': csv_path,
            'models_to_train': model_config['models'],
            'hyperparameter_tuning': model_config['hyperparameter_tuning'],
            'imputation_strategy': preprocessing_config['imputation_strategy'],
            'scaling_method': preprocessing_config['scaling_method'],
            'test_size': test_size,
            'experiment_name': f"{model_name}_{model_id[:8]}"
        }
        
        # Initialize training status
        training_status[job_id] = {
            'status': 'queued',
            'progress': 0,
            'message': 'Training job queued...',
            'model_id': model_id,
            'model_name': model_name
        }
        
        # Add job to queue
        training_queue.put(training_job)
        
        flash(f'Training started! Job ID: {job_id}', 'success')
        return redirect(url_for('training_status', job_id=job_id))
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/training_status/<job_id>')
def show_training_status(job_id):
    """Show training status page."""
    if job_id not in training_status:
        flash('Training job not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('training_status.html', 
                         job_id=job_id, 
                         status=training_status[job_id])

@app.route('/api/training_status/<job_id>')
def get_training_status(job_id):
    """API endpoint to get training status."""
    if job_id not in training_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(training_status[job_id])

@app.route('/model_details/<model_id>')
def model_details(model_id):
    """Show detailed model information."""
    if db is not None:  # Add this check
        model = db.models.find_one({'model_id': model_id})
        if not model:
            flash('Model not found', 'error')
            return redirect(url_for('index'))
        
        return render_template('model_details.html', model=model)
    else:
        flash('Database not available', 'error')
        return redirect(url_for('index'))

@app.route('/download/best_model/<model_id>')
def download_best_model(model_id):
    """Serve best model file for download."""
    if db is not None:  # Add this check
        model = db.models.find_one({'model_id': model_id})
        if not model or 'model_files' not in model:
            abort(404)
        
        best_model_path = model['model_files'].get('best_model')
        if not best_model_path or not os.path.exists(best_model_path):
            abort(404)
        
        return send_file(best_model_path, 
                        as_attachment=True, 
                        download_name=f"{model['name']}_best_model.pkl")
    else:
        abort(404)

@app.route('/download/all_models/<model_id>')
def download_all_models(model_id):
    """Serve all models file for download."""
    model = db.models.find_one({'model_id': model_id})
    if not model or 'model_files' not in model:
        abort(404)
    
    all_models_path = model['model_files'].get('all_models')
    if not all_models_path or not os.path.exists(all_models_path):
        abort(404)
    
    return send_file(all_models_path, 
                    as_attachment=True, 
                    download_name=f"{model['name']}_all_models.pkl")

@app.route('/download/preprocessors/<model_id>')
def download_preprocessors(model_id):
    """Serve preprocessors file for download."""
    model = db.models.find_one({'model_id': model_id})
    if not model or 'model_files' not in model:
        abort(404)
    
    preprocessors_path = model['model_files'].get('preprocessors')
    if not preprocessors_path or not os.path.exists(preprocessors_path):
        abort(404)
    
    return send_file(preprocessors_path, 
                    as_attachment=True, 
                    download_name=f"{model['name']}_preprocessors.pkl")

@app.route('/download/doc/<model_id>')
def download_doc(model_id):
    """Serve documentation file for download."""
    model = db.models.find_one({'model_id': model_id})
    if not model or not os.path.exists(model['document_file_path']):
        abort(404)
    
    return send_file(model['document_file_path'], 
                    as_attachment=True, 
                    download_name=f"{model['name']}_documentation.pdf")

@app.route('/like/<model_id>', methods=['POST'])
def like_model(model_id):
    """Handle model like action."""
    user_ip = request.remote_addr
    
    if db is not None:  # Add this check
        if has_user_liked(db, model_id, user_ip):
            return jsonify({'error': 'Already liked'}), 400
        
        try:
            save_like(db, model_id, user_ip)
            db.models.update_one({'model_id': model_id}, {'$inc': {'like_count': 1}})
            return jsonify({'success': True, 'message': 'Model liked successfully'})
        except Exception as e:
            logger.error(f"Error liking model: {e}")
            return jsonify({'error': 'Failed to like model'}), 500
    else:
        return jsonify({'error': 'Database not available'}), 500

@app.route('/api/models')
def api_models():
    """API endpoint to get all models."""
    if db is not None:  # Add this check
        models = get_all_models(db)
        # Convert ObjectId to string for JSON serialization
        for model in models:
            model['_id'] = str(model['_id'])
            if 'upload_date' in model:
                model['upload_date'] = model['upload_date'].isoformat()
        
        return jsonify(models)
    else:
        return jsonify([])

@app.route('/delete_model/<model_id>', methods=['POST'])
def delete_model(model_id):
    """Delete a model and its associated files."""
    if db is not None:  # Add this check
        try:
            model = db.models.find_one({'model_id': model_id})
            if not model:
                flash('Model not found', 'error')
                return redirect(url_for('index'))
            
            # Delete files
            files_to_delete = []
            if 'model_files' in model:
                files_to_delete.extend(model['model_files'].values())
            if 'document_file_path' in model:
                files_to_delete.append(model['document_file_path'])
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            # Delete from database
            db.models.delete_one({'model_id': model_id})
            db.likes.delete_many({'model_id': model_id})
            
            flash('Model deleted successfully', 'success')
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            flash('Error deleting model', 'error')
    else:
        flash('Database not available', 'error')
    
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    flash('Resource not found', 'error')
    return redirect(url_for('index'))

@app.errorhandler(413)
def file_too_large(e):
    flash('File too large. Maximum size is 50MB.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, threaded=True)