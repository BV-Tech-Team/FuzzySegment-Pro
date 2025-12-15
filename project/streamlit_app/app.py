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

st.set_page_config(page_title='FuzzySegment Pro', page_icon='🎯', layout='wide')

st.title('🎯 FuzzySegment Pro')
st.markdown('**Intelligent Customer Profiling using Fuzzy C-Means & Granular Computing**')
st.markdown('---')

# Sidebar configuration
st.sidebar.header('⚙️ Configuration')
uploaded = st.sidebar.file_uploader('Upload Preprocessed Customer CSV', type=['csv'])

if uploaded is None:
    default_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_customers.csv')
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        st.sidebar.success('Using default preprocessed dataset')
    else:
        st.warning('⚠️ No dataset found. Upload a CSV or run preprocessing first.')
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
st.sidebar.markdown('### Clustering Parameters')
n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 3)
fuzzifier = st.sidebar.slider('Fuzzifier (m)', 1.5, 3.0, 2.0, 0.1)
run_comparison = st.sidebar.checkbox('Compare with K-Means', value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header('📊 Dataset Overview')
    st.write(f'**Samples:** {len(X)} customers')
    st.write(f'**Features:** {len(selected_features)}')
    st.dataframe(df.head(10))

with col2:
    st.header('📈 Feature Statistics')
    st.dataframe(df[selected_features].describe())

st.markdown('---')

if st.button('🚀 Run Clustering', type='primary'):
    with st.spinner('Running Fuzzy C-Means...'):
        fcm = FuzzyCMeansWrapper(n_clusters=n_clusters, m=fuzzifier)
        fcm.fit(X)
        
        st.success('✓ Clustering Complete!')
        
        # Metrics
        st.header('📐 Fuzzy C-Means Metrics')
        metrics = evaluate_fuzzy_clustering(X, fcm.u, fcm.centers)
        
        metric_cols = st.columns(5)
        metric_cols[0].metric('PC', f"{metrics['PC']:.3f}")
        metric_cols[1].metric('MPC', f"{metrics['MPC']:.3f}")
        metric_cols[2].metric('PE', f"{metrics['PE']:.3f}")
        metric_cols[3].metric('XBI', f"{metrics['XBI']:.3f}")
        metric_cols[4].metric('FSI', f"{metrics['FSI']:.3f}")
        
        st.markdown('---')
        
        # Membership visualization
        st.header('🔍 Fuzzy Membership Matrix')
        n_show = min(50, len(X))
        
        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(fcm.u[:, :n_show], aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax.set_title(f'Membership Degrees (First {n_show} Customers)', fontsize=14)
        ax.set_xlabel('Customer Index')
        ax.set_ylabel('Cluster')
        ax.set_yticks(range(n_clusters))
        ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
        plt.colorbar(im, ax=ax, label='Membership Degree')
        st.pyplot(fig)
        
        # Sample customers
        st.header('👥 Sample Customer Profiles')
        sample_indices = np.random.choice(len(X), size=min(10, len(X)), replace=False)
        sample_data = []
        
        for idx in sample_indices:
            cust_id = customer_ids[idx]
            memberships = fcm.u[:, idx]
            sample_data.append({
                'Customer ID': cust_id,
                **{f'Cluster {i}': f'{m:.1%}' for i, m in enumerate(memberships)}
            })
        
        st.dataframe(pd.DataFrame(sample_data))
        
        # K-Means comparison
        if run_comparison:
            st.markdown('---')
            st.header('⚖️ Fuzzy C-Means vs K-Means')
            
            with st.spinner('Running K-Means for comparison...'):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_labels = kmeans.fit_predict(X)
                kmeans_silhouette = silhouette_score(X, kmeans_labels)
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.subheader('Fuzzy C-Means')
                    st.metric('Fuzzy Silhouette (FSI)', f"{metrics['FSI']:.3f}")
                    st.metric('Partition Coefficient (PC)', f"{metrics['PC']:.3f}")
                    max_memberships = np.max(fcm.u, axis=0)
                    multi_dim = np.sum(max_memberships < 0.7)
                    st.metric('Multi-dimensional Customers', f'{multi_dim} ({multi_dim/len(X):.1%})')
                
                with comp_col2:
                    st.subheader('K-Means')
                    st.metric('Silhouette Score', f"{kmeans_silhouette:.3f}")
                    st.metric('Assignment Type', 'Hard (100%)')
                    st.metric('Multi-dimensional Customers', '0 (0.0%)')
                
                # Side-by-side visualization
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Fuzzy memberships
                im1 = axes[0].imshow(fcm.u[:, :n_show], aspect='auto', cmap='YlOrRd', interpolation='nearest')
                axes[0].set_title('Fuzzy C-Means: Soft Assignments')
                axes[0].set_xlabel('Customer Index')
                axes[0].set_ylabel('Cluster')
                axes[0].set_yticks(range(n_clusters))
                plt.colorbar(im1, ax=axes[0], label='Membership')
                
                # K-Means hard assignments
                kmeans_matrix = np.zeros((n_clusters, n_show))
                for i in range(n_show):
                    kmeans_matrix[kmeans_labels[i], i] = 1
                im2 = axes[1].imshow(kmeans_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
                axes[1].set_title('K-Means: Hard Assignments')
                axes[1].set_xlabel('Customer Index')
                axes[1].set_ylabel('Cluster')
                axes[1].set_yticks(range(n_clusters))
                plt.colorbar(im2, ax=axes[1], label='Assignment')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                st.info('💡 **Insight:** Fuzzy C-Means reveals overlapping customer interests that K-Means misses!')

st.markdown('---')
st.markdown('**FuzzySegment Pro** | Powered by Fuzzy C-Means & Granular Computing')
