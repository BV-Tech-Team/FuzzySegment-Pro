import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.fuzzy_clustering import FuzzyCMeansWrapper
from src.metrics import evaluate_fuzzy_clustering

# Page configuration
st.set_page_config(
    page_title='FuzzySegment Pro',
    page_icon='chart',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Professional modern CSS with Font Awesome icons
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    :root {
        --primary: #1e40af;
        --secondary: #3b82f6;
        --accent: #60a5fa;
        --dark: #1e293b;
        --success: #10b981;
        --warning: #f59e0b;
    }
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(120deg, #1e40af 0%, #3b82f6 100%);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(30, 64, 175, 0.3);
    }
    
    .main-title {
        color: white !important;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-align: center;
    }
    
    .subtitle {
        color: #dbeafe;
        font-size: 1.4rem;
        text-align: center;
        margin-top: 1rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-top: 4px solid var(--secondary);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(59, 130, 246, 0.2);
    }
    
    .metric-icon {
        font-size: 2.5rem;
        color: var(--secondary);
        margin-bottom: 1rem;
    }
    
    .metric-label {
        font-size: 0.95rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        color: var(--dark);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(120deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(30, 64, 175, 0.4);
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(30, 64, 175, 0.5);
    }
    
    /* Section headers */
    .section-header {
        color: var(--dark);
        font-size: 2rem;
        font-weight: 800;
        margin: 2.5rem 0 1.5rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 4px solid var(--secondary);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border-left: 5px solid var(--accent);
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid var(--secondary);
        margin: 1.5rem 0;
    }
    
    .highlight-text {
        color: #1e40af;
        font-weight: 700;
        font-size: 1.15rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">FuzzySegment Pro</h1>
    <p class="subtitle">Intelligent Customer Profiling using Fuzzy C-Means & Granular Computing</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown('### <i class="fas fa-cog"></i> Configuration', unsafe_allow_html=True)
uploaded = st.sidebar.file_uploader('Upload Customer CSV', type=['csv'])

if uploaded is None:
    default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_customers.csv')
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.success('Using default preprocessed dataset')
    else:
        st.warning('No dataset found. Upload a CSV or run preprocessing first.')
        st.stop()
else:
    df = pd.read_csv(uploaded)

# Feature selection
id_cols = ['Customer ID', 'customer_id', 'ID', 'id']
id_col = None
for col in id_cols:
    if col in df.columns:
        id_col = col
        break

if id_col:
    feature_cols = [c for c in df.columns if c != id_col]
    customer_ids = df[id_col].values
else:
    feature_cols = list(df.columns)
    customer_ids = np.arange(len(df))

selected_features = st.sidebar.multiselect('Select Features', feature_cols, default=feature_cols)

if not selected_features:
    st.error('Please select at least one feature.')
    st.stop()

X = df[selected_features].values

# Clustering parameters
st.sidebar.markdown('### <i class="fas fa-sliders"></i> Clustering Parameters', unsafe_allow_html=True)
n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 3)
fuzzifier = st.sidebar.slider('Fuzzifier (m)', 1.5, 3.0, 2.0, 0.1)
run_comparison = st.sidebar.checkbox('Compare with K-Means', value=True)

# Dataset Overview
st.markdown('<div class="section-header"><i class="fas fa-database"></i> Dataset Overview</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-users metric-icon"></i>
        <div class="metric-label">Total Customers</div>
        <div class="metric-value">{len(X):,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-chart-line metric-icon"></i>
        <div class="metric-label">Features Selected</div>
        <div class="metric-value">{len(selected_features)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <i class="fas fa-sitemap metric-icon"></i>
        <div class="metric-label">Clusters</div>
        <div class="metric-value">{n_clusters}</div>
    </div>
    """, unsafe_allow_html=True)

# Data preview
with st.expander("View Dataset Preview", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    
with st.expander("View Feature Statistics", expanded=False):
    st.dataframe(df[selected_features].describe(), use_container_width=True)

st.markdown('<br>', unsafe_allow_html=True)

if st.button('Run Clustering Analysis', type='primary'):
    with st.spinner('Running Fuzzy C-Means clustering...'):
        fcm = FuzzyCMeansWrapper(n_clusters=n_clusters, m=fuzzifier)
        fcm.fit(X)
        
        st.markdown('<div class="success-box"><div class="highlight-text"><i class="fas fa-check-circle"></i> <span>Clustering Complete! Analysis results are displayed below.</span></div></div>', unsafe_allow_html=True)
        
        # Metrics
        st.markdown('<div class="section-header"><i class="fas fa-calculator"></i> Fuzzy C-Means Performance Metrics</div>', unsafe_allow_html=True)
        metrics = evaluate_fuzzy_clustering(X, fcm.u, fcm.centers)
        
        # Display metrics in cards
        metric_cols = st.columns(5)
        metric_info = [
            ('PC', metrics['PC'], 'Partition Coefficient', 'fa-circle-check'),
            ('MPC', metrics['MPC'], 'Modified PC', 'fa-chart-pie'),
            ('PE', metrics['PE'], 'Partition Entropy', 'fa-wave-square'),
            ('XBI', metrics['XBI'], 'Xie-Beni Index', 'fa-bullseye'),
            ('FSI', metrics['FSI'], 'Fuzzy Silhouette', 'fa-fingerprint')
        ]
        
        for col, (name, value, desc, icon) in zip(metric_cols, metric_info):
            with col:
                col.markdown(f"""
                <div class="metric-card">
                    <i class="fas {icon} metric-icon"></i>
                    <div class="metric-label">{name}</div>
                    <div class="metric-value" style="font-size: 1.8rem;">{value:.3f}</div>
                    <div style="font-size: 0.7rem; color: #64748b; margin-top: 0.5rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)
        
        # Membership visualization
        st.markdown('<div class="section-header"><i class="fas fa-grip"></i> Fuzzy Membership Matrix</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><i class="fas fa-info-circle"></i> The heatmap below shows fuzzy membership degrees for each customer across all clusters. Brighter colors indicate stronger membership.</div>', unsafe_allow_html=True)
        
        n_show = min(50, len(X))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(fcm.u[:, :n_show], aspect='auto', cmap='Blues', interpolation='nearest')
        ax.set_title(f'Membership Degrees (First {n_show} Customers)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Customer Index', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
        cbar = plt.colorbar(im, ax=ax, label='Membership Degree')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Sample customers
        st.markdown('<div class="section-header"><i class="fas fa-user-group"></i> Customer Profile Samples</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box"><i class="fas fa-info-circle"></i> Sample customers showing their fuzzy membership degrees across all clusters. Each row sums to 100%.</div>', unsafe_allow_html=True)
        
        sample_indices = np.random.choice(len(X), size=min(10, len(X)), replace=False)
        sample_data = []
        
        for idx in sample_indices:
            cust_id = customer_ids[idx]
            memberships = fcm.u[:, idx]
            row_data = {'Customer ID': cust_id}
            for i, m in enumerate(memberships):
                row_data[f'Cluster {i}'] = f'{m:.1%}'
            sample_data.append(row_data)
        
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
        
        # K-Means comparison
        if run_comparison:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><i class="fas fa-code-compare"></i> Fuzzy C-Means vs K-Means Comparison</div>', unsafe_allow_html=True)
            
            with st.spinner('Running K-Means for comparison...'):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X)
                kmeans_silhouette = silhouette_score(X, kmeans_labels)
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("""
                    <div style="background: #1e40af; 
                                padding: 2rem; border-radius: 12px; color: white; box-shadow: 0 8px 24px rgba(30, 64, 175, 0.3);">
                        <h3 style="margin-top: 0; color: white; display: flex; align-items: center; gap: 0.75rem;">
                            <i class="fas fa-layer-group"></i> Fuzzy C-Means
                        </h3>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Fuzzy Silhouette (FSI)</div>
                            <div style="font-size: 2rem; font-weight: 700;">{metrics['FSI']:.3f}</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Partition Coefficient (PC)</div>
                            <div style="font-size: 2rem; font-weight: 700;">{metrics['PC']:.3f}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    max_memberships = np.max(fcm.u, axis=0)
                    multi_dim = np.sum(max_memberships < 0.7)
                    st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Multi-dimensional Customers</div>
                            <div style="font-size: 2rem; font-weight: 700;">{multi_dim} ({multi_dim/len(X):.1%})</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with comp_col2:
                    st.markdown("""
                    <div style="background: #64748b; 
                                padding: 2rem; border-radius: 12px; color: white; box-shadow: 0 8px 24px rgba(100, 116, 139, 0.3);">
                        <h3 style="margin-top: 0; color: white; display: flex; align-items: center; gap: 0.75rem;">
                            <i class="fas fa-cube"></i> K-Means
                        </h3>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Silhouette Score</div>
                            <div style="font-size: 2rem; font-weight: 700;">{kmeans_silhouette:.3f}</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Assignment Type</div>
                            <div style="font-size: 2rem; font-weight: 700;">Hard (100%)</div>
                        </div>
                        <div style="margin: 1rem 0;">
                            <div style="font-size: 0.9rem; opacity: 0.9;">Multi-dimensional Customers</div>
                            <div style="font-size: 2rem; font-weight: 700;">0 (0.0%)</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Side-by-side visualization
                st.markdown('<br>', unsafe_allow_html=True)
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Fuzzy memberships
                im1 = axes[0].imshow(fcm.u[:, :n_show], aspect='auto', cmap='Blues', interpolation='nearest')
                axes[0].set_title('Fuzzy C-Means: Soft Assignments', fontsize=14, fontweight='bold', pad=15)
                axes[0].set_xlabel('Customer Index', fontsize=11)
                axes[0].set_ylabel('Cluster', fontsize=11)
                axes[0].set_yticks(range(n_clusters))
                plt.colorbar(im1, ax=axes[0], label='Membership')
                
                # K-Means hard assignments
                kmeans_matrix = np.zeros((n_clusters, n_show))
                for i in range(n_show):
                    kmeans_matrix[kmeans_labels[i], i] = 1
                im2 = axes[1].imshow(kmeans_matrix, aspect='auto', cmap='Blues', interpolation='nearest')
                axes[1].set_title('K-Means: Hard Assignments', fontsize=14, fontweight='bold', pad=15)
                axes[1].set_xlabel('Customer Index', fontsize=11)
                axes[1].set_ylabel('Cluster', fontsize=11)
                axes[1].set_yticks(range(n_clusters))
                plt.colorbar(im2, ax=axes[1], label='Assignment')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("""
                <div class="success-box">
                    <div class="highlight-text">
                        <i class="fas fa-lightbulb"></i>
                        <span>Key Insight: Fuzzy C-Means reveals overlapping customer interests that K-Means misses by forcing hard assignments!</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center; color: #64748b; padding: 2rem; border-top: 1px solid #e2e8f0;">' \
            '<strong>FuzzySegment Pro</strong> | Powered by BV Tech Team</div>', unsafe_allow_html=True)
