# app.py - Flask Web Application for Product Recommendation System
# Connects ML models with web interface for interactive recommendations

from flask import Flask, render_template, request, jsonify
import os
import sys
import logging
from datetime import datetime

# Add current directory to path to import our model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import get_recommendations, predict_sentiment, initialize_model
except ImportError as e:
    print(f"Error importing model: {e}")
    print("Make sure model.py and all pickle files are in the correct location")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           template_folder='flask_recommendation_app/templates',
           static_folder='flask_recommendation_app/static')

# Global variables
model_loaded = False

@app.before_first_request
def startup():
    """Initialize the ML model when the app starts."""
    global model_loaded
    try:
        initialize_model()
        model_loaded = True
        logger.info("✅ ML models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Error loading ML models: {str(e)}")
        model_loaded = False

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """
    API endpoint for getting product recommendations.
    Expects JSON with 'user_input' field.
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'ML models not loaded. Please try again later.',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        if not data or 'user_input' not in data:
            return jsonify({
                'error': 'Missing user_input in request',
                'status': 'error'
            }), 400
        
        user_input = data['user_input'].strip()
        if not user_input:
            return jsonify({
                'error': 'User input cannot be empty',
                'status': 'error'
            }), 400
        
        # Get number of recommendations (default: 5)
        n_recommendations = data.get('n_recommendations', 5)
        
        # Get recommendations from ML model
        result = get_recommendations(user_input, n_recommendations)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    """
    API endpoint for sentiment analysis only.
    Expects JSON with 'text' field.
    """
    try:
        if not model_loaded:
            return jsonify({
                'error': 'ML models not loaded. Please try again later.',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing text in request',
                'status': 'error'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': 'Text cannot be empty',
                'status': 'error'
            }), 400
        
        # Get sentiment prediction
        result = predict_sentiment(text)
        result['timestamp'] = datetime.now().isoformat()
        result['status'] = 'success'
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in sentiment endpoint: {str(e)}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy' if model_loaded else 'models_not_loaded',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info')
def model_info():
    """Get information about the loaded ML models."""
    try:
        if not model_loaded:
            return jsonify({
                'error': 'ML models not loaded',
                'status': 'error'
            }), 500
        
        model = initialize_model()
        info = model.get_model_info()
        info['timestamp'] = datetime.now().isoformat()
        
        return jsonify(info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': f'Error getting model info: {str(e)}',
            'status': 'error'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

# For development
if __name__ == '__main__':
    print("🚀 Starting Product Recommendation Flask App...")
    print("📊 Features:")
    print("   • Sentiment Analysis using Random Forest (94.20% accuracy)")
    print("   • Item-Based Collaborative Filtering")
    print("   • Interactive Web Interface")
    print("   • RESTful API Endpoints")
    
    # Check if models exist
    models_exist = all([
        os.path.exists('models/sentiment_model.pkl'),
        os.path.exists('models/vectorizer.pkl'),
        os.path.exists('models/item_cf_model.pkl'),
        os.path.exists('models/recommendation_matrices.pkl'),
        os.path.exists('models/sentiment_rec_system.pkl')
    ])
    
    if not models_exist:
        print("⚠️ Warning: Some model files are missing!")
        print("Make sure all .pkl files are in the 'models/' directory")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )