import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_loader import (load_results, load_metadata, load_test_data, load_preprocessing, 
                         load_shap_results, load_lime_results, 
                         load_explainability_summary, get_explainability_plots, 
                         get_shap_plots, get_lime_plots)
import warnings
import os
from PIL import Image
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="4-DOF Ship Motion Forecasting | AI-Powered Maritime Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ENHANCED CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
    }
    
    .sub-header {
        text-align: center;
        color: #4a5568;
        margin-bottom: 2.5rem;
        font-size: 1.3rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f7fafc;
        border-radius: 8px;
        font-weight: 600;
        font-size: 15px;
        color: #4a5568;
        transition: all 0.3s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #edf2f7;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    
    h2, h3 {
        color: #2d3748;
        font-weight: 600;
    }
    
    .safety-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .safety-warning {
        background: linear-gradient(135deg, #ff9f43 0%, #feca57 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
    
    .safety-normal {
        background: linear-gradient(135deg, #1dd1a1 0%, #10ac84 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SAFETY THRESHOLD CONFIGURATION
# ============================================================
SAFETY_THRESHOLDS = {
    'roll': {
        'threshold': 15.0,  # degrees
        'unit': '°',
        'description': 'Roll Alert: Stability Risk',
        'severity': 'high',
        'color': '#ff4444'
    },
    'yaw': {
        'threshold': 5.0,  # degrees/second
        'unit': '°/s',
        'description': 'Yaw Alert: Directional Instability',
        'severity': 'high',
        'color': '#ff8800'
    },
    'surge': {
        'threshold': 2.0,  # m/s²
        'unit': 'm/s²',
        'description': 'Surge Alert: Extreme Acceleration',
        'severity': 'medium',
        'color': '#ffaa00'
    },
    'sway': {
        'threshold': 1.0,  # m/s²
        'unit': 'm/s²',
        'description': 'Sway Alert: Lateral Movement Risk',
        'severity': 'medium',
        'color': '#ffcc00',
        'compound_condition': {'roll': 10.0}  # Additional condition
    }
}

def check_safety_thresholds(data):
    """
    Check if motion parameters exceed safety thresholds

    Parameters:
    -----------
    data : dict or pd.DataFrame
        Dictionary or DataFrame containing 'surge', 'sway', 'roll', 'yaw' values

    Returns:
    --------
    dict : Safety status for each DOF
    """
    if isinstance(data, pd.DataFrame):
        # Get the most recent values
        surge = abs(data['surge'].iloc[-1]) if 'surge' in data.columns else 0
        sway = abs(data['sway'].iloc[-1]) if 'sway' in data.columns else 0
        roll = abs(data['roll'].iloc[-1]) if 'roll' in data.columns else 0
        yaw = abs(data['yaw'].iloc[-1]) if 'yaw' in data.columns else 0
    else:
        surge = abs(data.get('surge', 0))
        sway = abs(data.get('sway', 0))
        roll = abs(data.get('roll', 0))
        yaw = abs(data.get('yaw', 0))

    safety_status = {}

    # Check Roll
    safety_status['roll'] = {
        'value': roll,
        'threshold': SAFETY_THRESHOLDS['roll']['threshold'],
        'exceeded': roll > SAFETY_THRESHOLDS['roll']['threshold'],
        'percentage': (roll / SAFETY_THRESHOLDS['roll']['threshold']) * 100
    }

    # Check Yaw
    safety_status['yaw'] = {
        'value': yaw,
        'threshold': SAFETY_THRESHOLDS['yaw']['threshold'],
        'exceeded': yaw > SAFETY_THRESHOLDS['yaw']['threshold'],
        'percentage': (yaw / SAFETY_THRESHOLDS['yaw']['threshold']) * 100
    }

    # Check Surge
    safety_status['surge'] = {
        'value': surge,
        'threshold': SAFETY_THRESHOLDS['surge']['threshold'],
        'exceeded': surge > SAFETY_THRESHOLDS['surge']['threshold'],
        'percentage': (surge / SAFETY_THRESHOLDS['surge']['threshold']) * 100
    }

    # Check Sway (with compound condition)
    sway_exceeded = sway > SAFETY_THRESHOLDS['sway']['threshold']
    compound_exceeded = roll > SAFETY_THRESHOLDS['sway']['compound_condition']['roll']

    safety_status['sway'] = {
        'value': sway,
        'threshold': SAFETY_THRESHOLDS['sway']['threshold'],
        'exceeded': sway_exceeded and compound_exceeded,
        'sway_only_exceeded': sway_exceeded,
        'compound_exceeded': compound_exceeded,
        'percentage': (sway / SAFETY_THRESHOLDS['sway']['threshold']) * 100
    }

    return safety_status

def display_safety_dashboard(safety_status):
    """Display safety threshold monitoring dashboard"""
    st.markdown("### Real-Time Safety Threshold Monitoring")

    # Check if any threshold is exceeded
    any_alert = any([status['exceeded'] for dof, status in safety_status.items()])

    if any_alert:
        st.markdown('<div class="safety-alert">SAFETY ALERT ACTIVE - One or more thresholds exceeded!</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="safety-normal">All motion parameters within safe operational limits</div>', unsafe_allow_html=True)

    # Display each DOF status
    cols = st.columns(4)

    dof_list = ['roll', 'yaw', 'surge', 'sway']

    for idx, dof in enumerate(dof_list):
        with cols[idx]:
            status = safety_status[dof]
            config = SAFETY_THRESHOLDS[dof]

            # Determine status color and symbol
            if status['exceeded']:
                status_text = "HIGH RISK"
                bg_color = "#ffe6e6"
                text_color = "#ff4444"
            elif status['percentage'] > 80:
                status_text = "WARNING"
                bg_color = "#fff9e6"
                text_color = "#ffaa00"
            else:
                status_text = "NORMAL"
                bg_color = "#e6ffe6"
                text_color = "#10ac84"

            # Display card
            st.markdown(f"""
            <div style="background-color: {bg_color}; padding: 15px; border-radius: 10px; border: 2px solid {config['color']};">
                <h4 style="color: {text_color}; margin: 0 0 10px 0;">{dof.upper()} - {status_text}</h4>
                <p style="font-size: 24px; font-weight: bold; margin: 5px 0;">
                    {status['value']:.3f} {config['unit']}
                </p>
                <p style="font-size: 12px; color: #666;">
                    Threshold: {status['threshold']:.1f} {config['unit']}
                </p>
                <div style="background-color: #ddd; border-radius: 5px; height: 8px; margin-top: 8px;">
                    <div style="background-color: {config['color']}; width: {min(status['percentage'], 100):.1f}%; height: 100%; border-radius: 5px;"></div>
                </div>
                <p style="font-size: 11px; margin-top: 5px;">
                    {status['percentage']:.1f}% of threshold
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Special compound condition for sway
            if dof == 'sway' and status.get('compound_exceeded'):
                if status['sway_only_exceeded']:
                    st.warning(f"Compound risk: Sway + Roll > 10°")

def plot_safety_thresholds_timeline(data, predictions=None):
    """Plot time series with safety threshold lines"""
    st.markdown("### Motion Parameters with Safety Thresholds")

    dof_list = ['surge', 'sway', 'roll', 'yaw']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, dof in enumerate(dof_list):
        ax = axes[idx]

        if dof in data.columns:
            # Plot actual data
            time_vals = np.arange(len(data))
            ax.plot(time_vals, data[dof], label='Actual', linewidth=2, alpha=0.8)

            # Plot predictions if available
            if predictions is not None and dof in predictions.columns:
                ax.plot(time_vals, predictions[dof], label='Predicted', 
                       linewidth=2, alpha=0.7, linestyle='--')

            # Add threshold lines
            threshold_val = SAFETY_THRESHOLDS[dof]['threshold']
            ax.axhline(y=threshold_val, color=SAFETY_THRESHOLDS[dof]['color'], 
                      linestyle='--', linewidth=2, label=f'Upper Threshold ({threshold_val})')
            ax.axhline(y=-threshold_val, color=SAFETY_THRESHOLDS[dof]['color'], 
                      linestyle='--', linewidth=2, label=f'Lower Threshold (-{threshold_val})')

            # Fill threshold zones
            ax.fill_between(time_vals, threshold_val, ax.get_ylim()[1], 
                          alpha=0.2, color='red', label='Danger Zone')
            ax.fill_between(time_vals, -threshold_val, ax.get_ylim()[0], 
                          alpha=0.2, color='red')

            # Styling
            ax.set_title(f'{dof.upper()} - {SAFETY_THRESHOLDS[dof]["description"]}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps', fontsize=10)
            ax.set_ylabel(f'{dof.capitalize()} ({SAFETY_THRESHOLDS[dof]["unit"]})', fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def calculate_safety_statistics(data):
    """Calculate safety statistics across time series"""
    st.markdown("### Safety Statistics Summary")

    stats_data = []

    for dof in ['surge', 'sway', 'roll', 'yaw']:
        if dof in data.columns:
            values = abs(data[dof])
            threshold = SAFETY_THRESHOLDS[dof]['threshold']

            violations = (values > threshold).sum()
            total_points = len(values)
            violation_percentage = (violations / total_points) * 100
            max_value = values.max()
            mean_value = values.mean()

            stats_data.append({
                'DOF': dof.upper(),
                'Threshold': f"{threshold} {SAFETY_THRESHOLDS[dof]['unit']}",
                'Max Value': f"{max_value:.3f}",
                'Mean Value': f"{mean_value:.3f}",
                'Violations': violations,
                'Violation %': f"{violation_percentage:.2f}%",
                'Safety Score': f"{100 - violation_percentage:.1f}%"
            })

    stats_df = pd.DataFrame(stats_data)

    # Display as styled table
    st.dataframe(stats_df, use_container_width=True)

    # Display safety score gauge
    col1, col2, col3 = st.columns(3)

    with col1:
        avg_safety = stats_df['Safety Score'].str.rstrip('%').astype(float).mean()
        st.metric("Overall Safety Score", f"{avg_safety:.1f}%", 
                 delta=f"{avg_safety - 95:.1f}% from target" if avg_safety < 95 else "Excellent")

    with col2:
        total_violations = stats_df['Violations'].sum()
        st.metric("Total Threshold Violations", total_violations)

    with col3:
        critical_dofs = sum([1 for _, row in stats_df.iterrows() 
                           if float(row['Violation %'].rstrip('%')) > 5])
        st.metric("Critical DOFs", f"{critical_dofs}/4")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_resource
def load_all_data():
    try:
        results = load_results()
        metadata = load_metadata()
        test_data = load_test_data()
        preprocessing = load_preprocessing()
        shap_results = load_shap_results()
        lime_results = load_lime_results()
        explainability_summary = load_explainability_summary()
        explainability_plots = get_explainability_plots()
        
        return (results, metadata, test_data, preprocessing, 
                shap_results, lime_results, explainability_summary, explainability_plots)
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None, None, None, None, None, None

with st.spinner("Initializing Advanced Maritime Analytics System..."):
    (results, metadata, test_data, preprocessing, 
     shap_results, lime_results, explainability_summary, explainability_plots) = load_all_data()

if results is None or metadata is None or test_data is None:
    st.error("Failed to load data files")
    st.stop()

# ============================================================
# HEADER
# ============================================================
st.markdown('<h1 class="main-header">4-DOF Ship Motion Forecasting</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced Deep Learning with Explainable AI for Maritime Safety</p>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style='text-align: center; padding: 20px 0; background: white; border-radius: 10px; margin-bottom: 20px;'>
    <h2 style='color: #667eea; margin: 0; font-size: 1.5rem;'>Dashboard Controls</h2>
</div>
""", unsafe_allow_html=True)

available_models = list(results.keys())
default_selection = ['Hermite-Hyperplane', 'PSO-BiLSTM', 'WPCA-DC-LSTM']
default_selection = [m for m in default_selection if m in available_models]

selected_models = st.sidebar.multiselect(
    "Select Models for Comparison",
    options=available_models,
    default=default_selection
)

signals = metadata['signals']
selected_dof = st.sidebar.selectbox("Degree of Freedom", options=signals, index=0)
dof_idx = signals.index(selected_dof)

max_samples = len(test_data['y_test'])
st.sidebar.markdown("### Visualization Settings")
sample_range = st.sidebar.slider("Sample Range", 0, max_samples-1, (0, min(300, max_samples-1)))

st.sidebar.markdown("---")
st.sidebar.markdown("### Display Options")
show_grid = st.sidebar.checkbox("Show Grid Lines", value=True)
show_uncertainty = st.sidebar.checkbox("Show Uncertainty Bounds", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Statistics")
col1, col2 = st.sidebar.columns(2)
col1.metric("Test Samples", max_samples)
col2.metric("Timesteps", metadata['timesteps'])
col1.metric("Features", metadata['n_features'])
col2.metric("Models", len(results))

# ============================================================
# MAIN TABS - UPDATED WITH RISK & SAFETY TAB
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Model Comparison", 
    "Performance Metrics", 
    "Detailed Analysis",
    "Uncertainty Analysis",
    "Risk & Safety Monitoring",
    "Explainability",
    "About & Documentation"
])

# ============================================================
# TAB 1: Model Comparison
# ============================================================
with tab1:
    st.header("Interactive Model Predictions Comparison")
    
    if not selected_models:
        st.warning("Please select at least one model from the sidebar")
    else:
        fig, ax = plt.subplots(figsize=(18, 8))
        fig.patch.set_facecolor('white')
        
        start_idx, end_idx = sample_range
        true_vals = test_data['y_test'][start_idx:end_idx, dof_idx]
        x_range = np.arange(len(true_vals))
        
        ax.plot(x_range, true_vals, 'k-', label='Ground Truth', linewidth=3, alpha=0.9, zorder=10)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(selected_models)))
        
        for idx, model_name in enumerate(selected_models):
            if model_name in results:
                pred = results[model_name]['ypred'][start_idx:end_idx]
                
                if pred.ndim > 1:
                    pred = pred[:, dof_idx]
                
                r2_score = results[model_name].get('r2', 0)
                ax.plot(x_range, pred, '-', color=colors[idx], 
                       label=f'{model_name} (R²={r2_score:.4f})', linewidth=2, alpha=0.8)
                
                if show_uncertainty and 'lower_bound' in results[model_name]:
                    lower = results[model_name]['lower_bound'][start_idx:end_idx]
                    upper = results[model_name]['upper_bound'][start_idx:end_idx]
                    
                    if lower.ndim > 1:
                        lower = lower[:, dof_idx]
                        upper = upper[:, dof_idx]
                    
                    ax.fill_between(x_range, lower, upper, alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{selected_dof} Value', fontsize=14, fontweight='bold')
        ax.set_title(f'Predictions for {selected_dof} Motion', fontsize=17, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.95, shadow=True)
        
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# TAB 2: Performance Metrics
# ============================================================
with tab2:
    st.header("Comprehensive Performance Metrics Analysis")
    
    metrics_data = []
    for model_name in available_models:
        if model_name in results:
            metrics_data.append({
                'Model': model_name,
                'RMSE': results[model_name].get('rmse', np.nan),
                'MAE': results[model_name].get('mae', np.nan),
                'R²': results[model_name].get('r2', np.nan)
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    df_metrics = df_metrics.sort_values('R²', ascending=False).reset_index(drop=True)
    
    st.subheader("Performance Summary Table")
    
    styled_df = df_metrics.style\
        .highlight_min(subset=['RMSE', 'MAE'], color='#32CD32')\
        .highlight_max(subset=['R²'], color='#228B22')\
        .format({'RMSE': '{:.6f}', 'MAE': '{:.6f}', 'R²': '{:.6f}'})\
        .background_gradient(subset=['R²'], cmap='Greens')
    
    st.dataframe(styled_df, use_container_width=True, height=300)
    
    st.subheader("Top Performing Models")
    col1, col2, col3 = st.columns(3)
    
    best_r2_model = df_metrics.iloc[0]
    lowest_rmse = df_metrics.loc[df_metrics['RMSE'].idxmin()]
    lowest_mae = df_metrics.loc[df_metrics['MAE'].idxmin()]
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 20px; border-radius: 12px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);'>
            <h4 style='color: white; margin: 0; font-size: 1rem; font-weight: 300;'>Best R² Score</h4>
            <h2 style='color: white; margin: 10px 0; font-size: 2rem; font-weight: 700;'>{best_r2_model['R²']:.6f}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>{best_r2_model['Model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 20px; border-radius: 12px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(240, 147, 251, 0.3);'>
            <h4 style='color: white; margin: 0; font-size: 1rem; font-weight: 300;'>Lowest RMSE</h4>
            <h2 style='color: white; margin: 10px 0; font-size: 2rem; font-weight: 700;'>{lowest_rmse['RMSE']:.6f}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>{lowest_rmse['Model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 20px; border-radius: 12px; text-align: center; 
                    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);'>
            <h4 style='color: white; margin: 0; font-size: 1rem; font-weight: 300;'>Lowest MAE</h4>
            <h2 style='color: white; margin: 10px 0; font-size: 2rem; font-weight: 700;'>{lowest_mae['MAE']:.6f}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>{lowest_mae['Model']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Comparative Metric Visualizations")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor('white')
    
    axes[0].barh(df_metrics['Model'], df_metrics['RMSE'], color='#667eea', edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('RMSE', fontsize=13, fontweight='bold')
    axes[0].set_title('Root Mean Square Error', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    axes[1].barh(df_metrics['Model'], df_metrics['MAE'], color='#f5576c', edgecolor='black', linewidth=1.5)
    axes[1].set_xlabel('MAE', fontsize=13, fontweight='bold')
    axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    axes[2].barh(df_metrics['Model'], df_metrics['R²'], color='#00f2fe', edgecolor='black', linewidth=1.5)
    axes[2].set_xlabel('R²', fontsize=13, fontweight='bold')
    axes[2].set_title('Coefficient of Determination', fontsize=14, fontweight='bold')
    axes[2].invert_yaxis()
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ============================================================
# TAB 3: Detailed Analysis
# ============================================================
with tab3:
    st.header("Detailed Model Analysis")
    
    selected_detail = st.selectbox("Select model for detailed analysis", 
                                   options=available_models)
    
    if selected_detail in results:
        res = results[selected_detail]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"{res['rmse']:.6f}")
        col2.metric("MAE", f"{res['mae']:.6f}")
        col3.metric("R²", f"{res['r2']:.6f}")
        
        if 'coverage' in res:
            col4.metric("Coverage", f"{res['coverage']:.2f}%")
        
        st.subheader("Predicted vs True Values for All DOFs")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.patch.set_facecolor('white')
        axes = axes.flatten()
        
        for dof_idx_plot, (ax, dof_name) in enumerate(zip(axes, signals)):
            y_true = res['ytrue']
            y_pred = res['ypred']
            
            if y_true.ndim > 1:
                y_true_dof = y_true[:, dof_idx_plot]
                y_pred_dof = y_pred[:, dof_idx_plot]
            else:
                y_true_dof = y_true
                y_pred_dof = y_pred
            
            ax.scatter(y_true_dof, y_pred_dof, alpha=0.5, s=40, color='steelblue', edgecolor='black', linewidth=0.5)
            
            min_val = min(y_true_dof.min(), y_pred_dof.min())
            max_val = max(y_true_dof.max(), y_pred_dof.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
            
            ss_res = np.sum((y_true_dof - y_pred_dof)**2)
            ss_tot = np.sum((y_true_dof - np.mean(y_true_dof))**2)
            r2_dof = 1 - (ss_res / ss_tot)
            
            ax.set_xlabel('True Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Values', fontsize=12, fontweight='bold')
            ax.set_title(f'{dof_name} (R²={r2_dof:.4f})', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{selected_detail} - Prediction Accuracy Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ============================================================
# TAB 4: Uncertainty Analysis
# ============================================================
with tab4:
    st.header("Uncertainty Quantification Analysis")
    
    uncertainty_models = [m for m in available_models if m in results and 'lower_bound' in results[m]]
    
    if not uncertainty_models:
        st.warning("No uncertainty quantification models found")
    else:
        selected_uncertainty = st.selectbox("Select uncertainty model", options=uncertainty_models)
        
        if selected_uncertainty in results:
            res = results[selected_uncertainty]
            
            if 'coverage' in res:
                col1, col2, col3 = st.columns(3)
                col1.metric("Coverage Percentage", f"{res['coverage']:.2f}%")
                col2.metric("Avg Interval Width", f"{res.get('avg_interval_width', 0):.6f}")
                col3.metric("R² Score", f"{res['r2']:.6f}")
            
            st.subheader("Predictions with Uncertainty Bounds")
            
            fig, ax = plt.subplots(figsize=(18, 8))
            fig.patch.set_facecolor('white')
            
            start_idx, end_idx = sample_range
            x_range = range(end_idx - start_idx)
            
            true_vals = res['ytrue'][start_idx:end_idx]
            pred_vals = res['ypred'][start_idx:end_idx]
            lower = res['lower_bound'][start_idx:end_idx]
            upper = res['upper_bound'][start_idx:end_idx]
            
            if true_vals.ndim > 1:
                true_vals = true_vals[:, dof_idx]
                pred_vals = pred_vals[:, dof_idx]
                lower = lower[:, dof_idx]
                upper = upper[:, dof_idx]
            
            ax.plot(x_range, true_vals, 'k-', label='Ground Truth', linewidth=3, zorder=10)
            ax.plot(x_range, pred_vals, 'b-', label='Prediction', linewidth=2.5)
            ax.fill_between(x_range, lower, upper, alpha=0.3, color='blue', label='95% Confidence Interval')
            
            ax.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
            ax.set_ylabel(f'{selected_dof}', fontsize=14, fontweight='bold')
            ax.set_title(f'{selected_uncertainty} - Uncertainty Quantification', fontsize=17, fontweight='bold')
            ax.legend(fontsize=13, framealpha=0.95, shadow=True)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ============================================================
# TAB 5: RISK & SAFETY MONITORING
# ============================================================
with tab5:
    st.header("Risk & Safety Monitoring Dashboard")
    
    st.markdown("""
    <div class='info-box'>
    <strong>Safety Threshold Monitoring:</strong>
    Real-time monitoring of ship motion parameters against established safety thresholds.
    Alerts are triggered when values exceed operational limits that could indicate dangerous conditions.
    </div>
    """, unsafe_allow_html=True)
    
    # Convert test data to DataFrame for safety analysis
    if 'y_test' in test_data and test_data['y_test'].ndim > 1:
        test_df = pd.DataFrame(test_data['y_test'], columns=signals)
        
        # Get current safety status
        current_safety_status = check_safety_thresholds(test_df)
        
        # Display safety dashboard
        display_safety_dashboard(current_safety_status)
        
        st.markdown("---")
        
        # Plot safety thresholds timeline
        plot_safety_thresholds_timeline(test_df)
        
        st.markdown("---")
        
        # Calculate and display safety statistics
        calculate_safety_statistics(test_df)
        
        # Additional Risk Analysis
        st.markdown("### Risk Assessment & Recommendations")
        
        # Calculate overall risk level
        total_violations = sum([1 for dof, status in current_safety_status.items() if status['exceeded']])
        
        if total_violations >= 2:
            risk_level = "HIGH RISK"
            risk_color = "red"
            recommendations = [
                "Immediate course correction required",
                "Reduce speed and stabilize vessel",
                "Monitor weather conditions closely",
                "Prepare for emergency procedures"
            ]
        elif total_violations == 1:
            risk_level = "MEDIUM RISK"
            risk_color = "orange"
            recommendations = [
                "Monitor the exceeded parameter closely",
                "Consider gradual course adjustment",
                "Check equipment and systems",
                "Increase monitoring frequency"
            ]
        else:
            risk_level = "LOW RISK"
            risk_color = "green"
            recommendations = [
                "Continue current operations",
                "Maintain regular monitoring",
                "Standard safety protocols sufficient"
            ]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div style='background-color: {risk_color}; color: white; padding: 20px; border-radius: 10px; text-align: center;'>
                <h3>Current Risk Level</h3>
                <h1>{risk_level}</h1>
                <p>Threshold Violations: {total_violations}/4</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Recommended Actions")
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        
        # Historical Safety Trends
        st.markdown("### Historical Safety Trends")
        
        # Calculate rolling safety scores
        window_size = min(50, len(test_df))
        safety_scores = []
        
        for dof in signals:
            if dof in test_df.columns:
                values = abs(test_df[dof])
                threshold = SAFETY_THRESHOLDS.get(dof, {}).get('threshold', 1.0)
                safe_points = (values <= threshold).astype(int)
                rolling_safe = safe_points.rolling(window=window_size).mean() * 100
                safety_scores.append(rolling_safe)
        
        if safety_scores:
            avg_safety = pd.concat(safety_scores, axis=1).mean(axis=1)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(avg_safety.index, avg_safety.values, linewidth=2, color='#667eea')
            ax.axhline(y=95, color='red', linestyle='--', label='Safety Target (95%)')
            ax.fill_between(avg_safety.index, avg_safety.values, 95, 
                          where=(avg_safety.values >= 95), alpha=0.3, color='green')
            ax.fill_between(avg_safety.index, avg_safety.values, 95, 
                          where=(avg_safety.values < 95), alpha=0.3, color='red')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Safety Score (%)')
            ax.set_title('Rolling Average Safety Score (50-sample window)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.warning("Test data not available in required format for safety analysis")

# ============================================================
# TAB 6: EXPLAINABILITY
# ============================================================
with tab6:
    st.header("Model Explainability: SHAP & LIME Analysis")
    
    st.markdown("""
    <div class='info-box'>
    <strong>Understanding Model Decisions:</strong>
    <ul>
        <li><strong>SHAP:</strong> Global feature importance using Shapley values from game theory</li>
        <li><strong>LIME:</strong> Local explanations for individual predictions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    has_shap = shap_results is not None and len(shap_results) > 0
    has_lime = lime_results is not None and len(lime_results) > 0
    has_plots = len(explainability_plots) > 0
    
    if not has_shap and not has_lime and not has_plots:
        st.warning("No explainability data found")
        st.info("Run SHAP/LIME analysis in your notebook and save results")
    else:
        explainability_subtabs = st.tabs(["SHAP Analysis", "LIME Analysis", "All Plots"])
        
        # SHAP Sub-tab
        with explainability_subtabs[0]:
            st.subheader("SHAP Feature Importance Analysis")
            
            if has_shap:
                shap_models_available = [m for m in shap_results.keys() if 'herman' not in m.lower()]
                
                if shap_models_available:
                    shap_model = st.selectbox("Select model for SHAP analysis", options=shap_models_available)
                    
                    if shap_model in shap_results:
                        shap_data = shap_results[shap_model]
                        
                        st.markdown(f"### SHAP Analysis: {shap_model}")
                        
                        if 'values' in shap_data:
                            shap_vals = shap_data['values']
                            
                            if isinstance(shap_vals, list):
                                shap_vals_main = shap_vals[0]
                                st.info("Multi-output model detected - using first output")
                            else:
                                shap_vals_main = shap_vals
                            
                            col1, col2, col3 = st.columns(3)
                            col1.metric("SHAP Values Shape", str(shap_vals_main.shape))
                            col2.metric("Mean |SHAP|", f"{np.mean(np.abs(shap_vals_main)):.4f}")
                            col3.metric("Max |SHAP|", f"{np.max(np.abs(shap_vals_main)):.4f}")
                            
                            st.markdown("#### Top 10 Most Important Features")
                            
                            mean_abs_shap = np.mean(np.abs(shap_vals_main), axis=0)
                            if mean_abs_shap.ndim > 1:
                                mean_abs_shap = mean_abs_shap.flatten()
                            
                            top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
                            top_values = mean_abs_shap[top_indices]
                            
                            fig, ax = plt.subplots(figsize=(12, 6))
                            fig.patch.set_facecolor('white')
                            ax.barh(range(10), top_values, color='steelblue', edgecolor='black', linewidth=1.5)
                            ax.set_yticks(range(10))
                            ax.set_yticklabels([f'Feature {i}' for i in top_indices])
                            ax.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
                            ax.set_title(f'Top 10 Features - {shap_model}', fontsize=14, fontweight='bold')
                            ax.invert_yaxis()
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                            
                            st.markdown("#### Generated SHAP Visualizations")
                            shap_plot_files = [p for p in explainability_plots if 'shap' in p.lower() and shap_model.replace(' ', '_').replace('-', '_').lower() in p.lower()]
                            
                            if shap_plot_files:
                                cols = st.columns(2)
                                for idx, plot_file in enumerate(shap_plot_files[:4]):
                                    with cols[idx % 2]:
                                        try:
                                            img = Image.open(f"explainability_plots/{plot_file}")
                                            st.image(img, caption=plot_file, use_container_width=True)
                                        except:
                                            st.error(f"Error loading {plot_file}")
                else:
                    st.info("No SHAP models available (Herman-Hyperplane excluded)")
            else:
                st.info("No SHAP data available")
        
        # LIME Sub-tab
        with explainability_subtabs[1]:
            st.subheader("LIME Local Explanations")
            
            if has_lime:
                lime_models_available = [m for m in lime_results.keys() if 'herman' not in m.lower()]
                
                if lime_models_available:
                    lime_model = st.selectbox("Select model for LIME analysis", options=lime_models_available)
                    
                    if lime_model in lime_results:
                        lime_data = lime_results[lime_model]
                        
                        st.markdown(f"### LIME Analysis: {lime_model}")
                        
                        if 'explanations' in lime_data:
                            explanations = lime_data['explanations']
                            st.metric("Number of Explanations Generated", len(explanations))
                            
                            st.markdown("#### LIME Explanation Visualizations")
                            lime_plot_files = [p for p in explainability_plots if 'lime' in p.lower() and lime_model.replace(' ', '_').replace('-', '_').lower() in p.lower()]
                            
                            if lime_plot_files:
                                agg_plots = [p for p in lime_plot_files if 'aggregated' in p.lower()]
                                if agg_plots:
                                    try:
                                        img = Image.open(f"explainability_plots/{agg_plots[0]}")
                                        st.image(img, caption="Aggregated Feature Importance", use_container_width=True)
                                    except:
                                        pass
                                
                                st.markdown("#### Individual Sample Explanations")
                                sample_plots = [p for p in lime_plot_files if 'sample' in p.lower()]
                                
                                cols = st.columns(2)
                                for idx, plot_file in enumerate(sample_plots[:6]):
                                    with cols[idx % 2]:
                                        try:
                                            img = Image.open(f"explainability_plots/{plot_file}")
                                            st.image(img, caption=plot_file, use_container_width=True)
                                        except:
                                            st.error(f"Error loading {plot_file}")
                else:
                    st.info("No LIME models available (Herman-Hyperplane excluded)")
            else:
                st.info("No LIME data available")
        
        # All Plots Sub-tab
        with explainability_subtabs[2]:
            st.subheader("All Explainability Visualizations")
            
            if has_plots:
                plot_type = st.selectbox("Filter plots by type", options=["All", "SHAP", "LIME"])
                
                filtered_plots = [p for p in explainability_plots if 'herman' not in p.lower()]
                
                if plot_type == "SHAP":
                    filtered_plots = [p for p in filtered_plots if 'shap' in p.lower()]
                elif plot_type == "LIME":
                    filtered_plots = [p for p in filtered_plots if 'lime' in p.lower()]
                
                if filtered_plots:
                    cols = st.columns(2)
                    for i, plot_file in enumerate(filtered_plots):
                        with cols[i % 2]:
                            try:
                                img = Image.open(f"explainability_plots/{plot_file}")
                                st.image(img, caption=plot_file, use_container_width=True)
                            except:
                                st.error(f"Error: {plot_file}")
                else:
                    st.info("No plots found")
            else:
                st.info("No plots found")

# ============================================================
# TAB 7: About & Documentation
# ============================================================
with tab7:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>About This Project</h1>
        <p style='color: rgba(255,255,255,0.9); text-align: center; margin-top: 10px; font-size: 1.1rem;'>
            AI-Powered Maritime Safety & Motion Prediction System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("### Project Overview")
    st.markdown("""
    <div class='info-box'>
    This comprehensive dashboard presents an advanced 4-Degree-of-Freedom (4-DOF) ship motion forecasting system
    that leverages cutting-edge deep learning techniques combined with explainable AI methodologies. 
    The system is designed to predict ship movements with high accuracy while providing transparency 
    through SHAP and LIME interpretability frameworks.
    
    **Key Features:**
    <ul>
        <li>Real-time prediction of ship motion across four critical axes</li>
        <li>Uncertainty quantification for risk assessment</li>
        <li>Explainable AI integration for model transparency</li>
        <li>Comparative analysis of six state-of-the-art deep learning architectures</li>
        <li>Interactive visualization and analysis tools</li>
        <li><strong>NEW:</strong> Real-time safety threshold monitoring and risk assessment</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Safety Thresholds Information
    st.markdown("### Safety Threshold Configuration")
    
    safety_df = pd.DataFrame([
        {
            'DOF': dof.upper(),
            'Threshold': f"{config['threshold']} {config['unit']}",
            'Description': config['description'],
            'Severity': config['severity'].upper(),
            'Color': config['color']
        }
        for dof, config in SAFETY_THRESHOLDS.items()
    ])
    
    st.dataframe(safety_df, use_container_width=True)
    
    # Degrees of Freedom Section
    st.markdown("### Understanding Ship Motion: Four Degrees of Freedom")
    
    cols = st.columns(4)
    dof_details = [
        ('Surge', 'Forward/backward translational motion along the longitudinal axis', '#667eea'),
        ('Sway', 'Side-to-side translational motion along the transverse axis', '#f5576c'),
        ('Roll', 'Rotational motion about the longitudinal axis', '#4facfe'),
        ('Yaw', 'Rotational motion about the vertical axis', '#764ba2')
    ]
    
    for col, (name, desc, color) in zip(cols, dof_details):
        with col:
            st.markdown(f"""
            <div style='background: white; padding: 20px; border-radius: 12px; text-align: center; 
                        border-top: 4px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                        height: 180px; display: flex; flex-direction: column; justify-content: center;'>
                <h3 style='color: {color}; margin: 0 0 10px 0; font-size: 1.3rem;'>{name}</h3>
                <p style='font-size: 0.9rem; color: #4a5568; line-height: 1.5; margin: 0;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Model Architecture Section
    st.markdown("### Implemented Deep Learning Architectures")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        **1. Hermite-Hyperplane (Best Performing - R²: 0.8444)**
        - Advanced dual hyperplane architecture with SVR-based prediction intervals
        - Robust uncertainty quantification capabilities
        - Optimal for complex non-linear maritime patterns
        
        **2. WPCA-DC-LSTM (Weighted PCA Dual-Channel)**
        - Weighted Principal Component Analysis for dimensionality reduction
        - Dual-channel LSTM architecture for parallel feature processing
        - Enhanced temporal dependency capture
        
        **3. PSO-BiLSTM (Particle Swarm Optimized Bidirectional LSTM)**
        - Bidirectional LSTM for forward and backward temporal processing
        - Particle Swarm Optimization for hyperparameter tuning
        - Superior performance on long-term dependencies
        
        **4. PCA-LSTM (Principal Component Analysis LSTM)**
        - Traditional PCA for feature dimensionality reduction
        - Dual-input LSTM architecture for reduced computational complexity
        
        **5. Conventional LSTM (Baseline Model)**
        - Standard 3-layer LSTM architecture for sequential processing
        - Benchmark for performance comparison
        
        **6. ADPSO-BiLSTM (Adaptive Particle Swarm Optimization)**
        - Self-tuning learning parameters with dynamic optimization
        - Improved convergence characteristics
        """)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    padding: 20px; border-radius: 12px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; margin-top: 0;'>Performance Metrics</h4>
            <p style='margin: 10px 0;'><strong>RMSE:</strong> Root Mean Square Error</p>
            <p style='margin: 10px 0;'><strong>MAE:</strong> Mean Absolute Error</p>
            <p style='margin: 10px 0;'><strong>R²:</strong> Coefficient of Determination</p>
            <p style='margin: 10px 0;'><strong>Coverage:</strong> Prediction Interval Coverage</p>
            <p style='margin: 10px 0;'><strong>Safety Score:</strong> Threshold Compliance Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Technical Stack
    st.markdown("### Technical Implementation Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #667eea;'>
            <h4 style='color: #667eea; margin-top: 0;'>Deep Learning</h4>
            <p>TensorFlow 2.x</p>
            <p>Keras API</p>
            <p>Scikit-learn</p>
            <p>NumPy & SciPy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #f5576c;'>
            <h4 style='color: #f5576c; margin-top: 0;'>Visualization</h4>
            <p>Streamlit</p>
            <p>Matplotlib</p>
            <p>Seaborn</p>
            <p>Plotly</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='background: white; padding: 15px; border-radius: 10px; border-left: 4px solid #4facfe;'>
            <h4 style='color: #4facfe; margin-top: 0;'>Explainability & Safety</h4>
            <p>SHAP Library</p>
            <p>LIME Library</p>
            <p>Custom Safety Monitoring</p>
            <p>Risk Assessment Algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Information
    st.markdown("### Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training Samples", metadata.get('train_samples', 'N/A'))
    with col2:
        st.metric("Testing Samples", metadata.get('test_samples', 'N/A'))
    with col3:
        st.metric("Input Features", metadata.get('n_features', 'N/A'))
    with col4:
        st.metric("Temporal Steps", metadata.get('timesteps', 'N/A'))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Applications Section
    st.markdown("### Real-World Maritime Applications")
    st.markdown("""
    <div class='info-box'>
    <strong>Industry Impact:</strong>
    <ul style='line-height: 1.8; margin-top: 10px;'>
        <li><strong>Navigation Safety:</strong> Real-time motion prediction for safer vessel operations</li>
        <li><strong>Route Optimization:</strong> Weather-adaptive routing based on predicted behavior</li>
        <li><strong>Structural Monitoring:</strong> Early detection of abnormal motion patterns</li>
        <li><strong>Crew Safety:</strong> Predictive alerts for hazardous conditions</li>
        <li><strong>Fuel Efficiency:</strong> Optimized speed and heading recommendations</li>
        <li><strong>Cargo Protection:</strong> Motion-based securing recommendations</li>
        <li><strong>Risk Mitigation:</strong> Proactive safety threshold monitoring and alerts</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 20px; border-radius: 12px; text-align: center;'>
        <p style='margin: 0; color: #4a5568; font-size: 0.95rem;'>
            <strong>Project Type:</strong> Mini Project | 
            <strong>Domain:</strong> Maritime AI & Deep Learning | 
            <strong>Year:</strong> 2025
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PROFESSIONAL FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; text-align: center; padding: 40px; border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);'>
    <h3 style='margin: 0; font-size: 26px; font-weight: 700;'>4-DOF Ship Motion Forecasting System</h3>
    <div style='width: 100px; height: 3px; background: white; margin: 15px auto; border-radius: 2px;'></div>
    <p style='margin: 15px 0 0 0; font-size: 16px; opacity: 0.95;'>
        AI-Powered Maritime Safety & Motion Prediction System
    </p>
</div>
""", unsafe_allow_html=True)