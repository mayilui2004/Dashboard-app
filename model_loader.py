import pickle
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

try:
    from tensorflow.keras.models import load_model
    KERAS_AVAILABLE = True
except ImportError:
    print("Warning: TensorFlow/Keras not available")
    KERAS_AVAILABLE = False

# Base directories
MODELS_DIR = "models"
DATA_DIR = "streamlit_data"
PLOTS_DIR = "explainability_plots"

# ============================================================
# KERAS MODEL LOADER
# ============================================================
def load_keras_model(model_name):
    """Load a Keras model"""
    if not KERAS_AVAILABLE:
        return None
    
    try:
        clean_name = model_name.replace(' ', '_').replace('-', '_')
        
        # Try .keras format
        keras_path = os.path.join(MODELS_DIR, f"{clean_name}.keras")
        if os.path.exists(keras_path):
            return load_model(keras_path)
        
        # Try .h5 format
        h5_path = os.path.join(MODELS_DIR, f"{clean_name}.h5")
        if os.path.exists(h5_path):
            return load_model(h5_path)
        
        return None
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

# ============================================================
# DATA LOADERS
# ============================================================
def load_results():
    """Load results dictionary"""
    try:
        results_path = os.path.join(DATA_DIR, 'results.pkl')
        if not os.path.exists(results_path):
            return None
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def load_test_data():
    """Load test dataset"""
    try:
        test_data_path = os.path.join(DATA_DIR, 'test_data.pkl')
        if not os.path.exists(test_data_path):
            return None
        
        with open(test_data_path, 'rb') as f:
            test_data = pickle.load(f)
        return test_data
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

def load_preprocessing():
    """Load preprocessing objects"""
    try:
        preprocessing_path = os.path.join(DATA_DIR, 'preprocessing.pkl')
        if not os.path.exists(preprocessing_path):
            return None
        
        with open(preprocessing_path, 'rb') as f:
            preprocessing = pickle.load(f)
        return preprocessing
    except Exception as e:
        print(f"Error loading preprocessing: {e}")
        return None

def load_metadata():
    """Load metadata"""
    try:
        metadata_path = os.path.join(DATA_DIR, 'metadata.pkl')
        if not os.path.exists(metadata_path):
            return None
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None

# ============================================================
# EXPLAINABILITY LOADERS (NEW)
# ============================================================
def load_shap_results():
    """Load SHAP analysis results"""
    try:
        shap_path = os.path.join(DATA_DIR, 'shap_results.pkl')
        if os.path.exists(shap_path):
            with open(shap_path, 'rb') as f:
                shap_results = pickle.load(f)
            print(f"Loaded SHAP results for {len(shap_results)} models")
            return shap_results
        else:
            print("SHAP results file not found")
            return None
    except Exception as e:
        print(f"Error loading SHAP results: {e}")
        return None

def load_lime_results():
    """Load LIME analysis results"""
    try:
        lime_path = os.path.join(DATA_DIR, 'lime_results.pkl')
        if os.path.exists(lime_path):
            with open(lime_path, 'rb') as f:
                lime_results = pickle.load(f)
            print(f"Loaded LIME results for {len(lime_results)} models")
            return lime_results
        else:
            print("LIME results file not found")
            return None
    except Exception as e:
        print(f"Error loading LIME results: {e}")
        return None

def load_explainability_summary():
    """Load explainability summary"""
    try:
        summary_path = os.path.join(DATA_DIR, 'explainability_summary.pkl')
        if os.path.exists(summary_path):
            with open(summary_path, 'rb') as f:
                summary = pickle.load(f)
            return summary
        return None
    except Exception as e:
        print(f"Error loading explainability summary: {e}")
        return None

def get_explainability_plots():
    """Get list of saved explainability plots"""
    if not os.path.exists(PLOTS_DIR):
        return []
    
    plot_files = [f for f in os.listdir(PLOTS_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(plot_files)

def get_shap_plots():
    """Get SHAP plots only"""
    all_plots = get_explainability_plots()
    return [p for p in all_plots if 'shap' in p.lower()]

def get_lime_plots():
    """Get LIME plots only"""
    all_plots = get_explainability_plots()
    return [p for p in all_plots if 'lime' in p.lower()]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def check_files():
    """Check if all required files exist"""
    required_files = {
        'results': os.path.join(DATA_DIR, 'results.pkl'),
        'test_data': os.path.join(DATA_DIR, 'test_data.pkl'),
        'metadata': os.path.join(DATA_DIR, 'metadata.pkl'),
        'preprocessing': os.path.join(DATA_DIR, 'preprocessing.pkl'),
        'shap_results': os.path.join(DATA_DIR, 'shap_results.pkl'),
        'lime_results': os.path.join(DATA_DIR, 'lime_results.pkl'),
    }
    
    status = {}
    print("\nChecking required files:")
    print("="*60)
    
    for name, path in required_files.items():
        exists = os.path.exists(path)
        status[name] = exists
        status_symbol = "✓" if exists else "✗"
        print(f"{status_symbol} {name:20s}: {path}")
    
    # Check directories
    models_exist = os.path.exists(MODELS_DIR)
    plots_exist = os.path.exists(PLOTS_DIR)
    
    status['models_dir'] = models_exist
    status['plots_dir'] = plots_exist
    
    print(f"{'✓' if models_exist else '✗'} {'models directory':20s}: {MODELS_DIR}")
    print(f"{'✓' if plots_exist else '✗'} {'plots directory':20s}: {PLOTS_DIR}")
    
    if plots_exist:
        plot_files = get_explainability_plots()
        print(f"  Found {len(plot_files)} explainability plots")
    
    print("="*60)
    
    return status

def load_all():
    """Load all data at once"""
    print("\n" + "="*60)
    print("Loading all data for Streamlit dashboard")
    print("="*60)
    
    results = load_results()
    metadata = load_metadata()
    test_data = load_test_data()
    preprocessing = load_preprocessing()
    shap_results = load_shap_results()
    lime_results = load_lime_results()
    explainability_summary = load_explainability_summary()
    explainability_plots = get_explainability_plots()
    
    print("\nLoad Summary:")
    print(f"  Results:       {'✓' if results is not None else '✗'}")
    print(f"  Metadata:      {'✓' if metadata is not None else '✗'}")
    print(f"  Test Data:     {'✓' if test_data is not None else '✗'}")
    print(f"  Preprocessing: {'✓' if preprocessing is not None else '✗'}")
    print(f"  SHAP Results:  {'✓' if shap_results is not None else '✗'}")
    print(f"  LIME Results:  {'✓' if lime_results is not None else '✗'}")
    print(f"  Plots:         {len(explainability_plots)} files")
    print("="*60)
    
    return (results, metadata, test_data, preprocessing, 
            shap_results, lime_results, explainability_summary, explainability_plots)

# ============================================================
# MAIN (for testing)
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*15 + "MODEL LOADER TEST (WITH EXPLAINABILITY)")
    print("="*70)
    
    status = check_files()
    
    if all(status.values()):
        print("\n✓ All files found")
        print("\nAttempting to load data...")
        
        data = load_all()
        
        results, metadata, test_data, preprocessing, shap_results, lime_results, summary, plots = data
        
        if results and metadata and test_data:
            print("\n✓ Core data loaded successfully!")
            print(f"  Models: {list(results.keys())}")
            print(f"  DOFs: {metadata['signals']}")
            print(f"  Test samples: {test_data['y_test'].shape[0]}")
            
            if shap_results:
                print(f"\n✓ SHAP results loaded!")
                print(f"  Models with SHAP: {list(shap_results.keys())}")
            
            if lime_results:
                print(f"\n✓ LIME results loaded!")
                print(f"  Models with LIME: {list(lime_results.keys())}")
            
            if plots:
                shap_plots = [p for p in plots if 'shap' in p.lower()]
                lime_plots = [p for p in plots if 'lime' in p.lower()]
                print(f"\n✓ Explainability plots found!")
                print(f"  SHAP plots: {len(shap_plots)}")
                print(f"  LIME plots: {len(lime_plots)}")
        else:
            print("\n✗ Some core data failed to load")
    else:
        print("\n✗ Some required files are missing")
    
    print("\n" + "="*70)
