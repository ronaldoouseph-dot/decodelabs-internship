import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors

# Set a cohesive dark/neon palette for styling
CLASS_COLORS = {
    0: '#6C5CE7',  # Setosa: Purple-blue
    1: '#00CEC9',  # Versicolor: Teal
    2: '#E84393'   # Virginica: Pink/rose
}
CLASS_COLORS_STR = {
    'setosa': '#6C5CE7',
    'versicolor': '#00CEC9',
    'virginica': '#E84393'
}

def plot_interactive_scatter(df, x_col, y_col):
    """
    Creates an interactive scatter plot of two selected features.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col, color='Species',
        color_discrete_map={'setosa': CLASS_COLORS[0], 'versicolor': CLASS_COLORS[1], 'virginica': CLASS_COLORS[2]},
        title=f"{x_col} vs {y_col} Distribution",
        labels={x_col: x_col, y_col: y_col, 'Species': 'Species Class'},
        hover_data=['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)']
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif"),
        title_font=dict(size=18, color="#FFFFFF"),
        xaxis=dict(gridcolor="#2D3436", title_font=dict(color="#B2BEC3")),
        yaxis=dict(gridcolor="#2D3436", title_font=dict(color="#B2BEC3")),
        legend=dict(font=dict(color="#FFFFFF"))
    )
    return fig

def plot_scaling_comparison(raw_df, scaled_df, feature_name):
    """
    Plots a before-and-after histogram comparison for a given feature.
    """
    fig = go.Figure()
    
    # Raw feature distribution
    fig.add_trace(go.Histogram(
        x=raw_df[feature_name],
        name='Raw Data',
        xbins=dict(size=0.2),
        marker_color='#FF7675',
        opacity=0.75
    ))
    
    # Scaled feature distribution
    fig.add_trace(go.Histogram(
        x=scaled_df[feature_name],
        name='Standard Scaled Data',
        xbins=dict(size=0.2),
        marker_color='#55EFC4',
        opacity=0.75
    ))
    
    fig.update_layout(
        barmode='overlay',
        title=f"Scaling Effect on '{feature_name}' (Raw vs Standardized)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif"),
        xaxis=dict(title="Value", gridcolor="#2D3436"),
        yaxis=dict(title="Frequency", gridcolor="#2D3436")
    )
    return fig

def plot_train_test_split_dist(y_train, y_test, class_names):
    """
    Plots a bar chart comparing the class distribution in the train and test splits.
    """
    train_counts = np.bincount(y_train, minlength=3)
    test_counts = np.bincount(y_test, minlength=3)
    
    fig = go.Figure(data=[
        go.Bar(name='Training Set (80%)', x=[c.capitalize() for c in class_names], y=train_counts, marker_color='#6C5CE7'),
        go.Bar(name='Testing Set (20%)', x=[c.capitalize() for c in class_names], y=test_counts, marker_color='#E84393')
    ])
    
    fig.update_layout(
        barmode='group',
        title="Class Distribution in Splits (Stratified)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif"),
        xaxis=dict(title="Species", gridcolor="#2D3436"),
        yaxis=dict(title="Count", gridcolor="#2D3436")
    )
    return fig

def plot_k_error_rates(k_values, error_rates, selected_k=5):
    """
    Plots error rate vs K values, highlighting the selected K.
    """
    selected_idx = k_values.index(selected_k)
    selected_error = error_rates[selected_idx]
    
    fig = go.Figure()
    
    # Error rate curve
    fig.add_trace(go.Scatter(
        x=k_values, y=error_rates,
        mode='lines+markers',
        name='Error Rate',
        line=dict(color='#00CEC9', width=3),
        marker=dict(size=8, color='#00CEC9')
    ))
    
    # Highlight selected K
    fig.add_trace(go.Scatter(
        x=[selected_k], y=[selected_error],
        mode='markers',
        name=f'Selected K = {selected_k}',
        marker=dict(size=14, color='#E84393', symbol='star')
    ))
    
    fig.update_layout(
        title="Model Tuning: K-Value vs Error Rate (Elbow Method)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif"),
        xaxis=dict(title="Number of Neighbors (K)", dtick=1, gridcolor="#2D3436"),
        yaxis=dict(title="Error Rate (1 - Accuracy)", gridcolor="#2D3436")
    )
    return fig

def plot_confusion_matrix_heatmap(cm, class_names):
    """
    Renders the confusion matrix as a high-end heatmap.
    """
    labels = [c.capitalize() for c in class_names]
    
    # Custom hovertext to explain matrix details
    hover_text = []
    for i in range(len(labels)):
        row = []
        for j in range(len(labels)):
            actual = labels[i]
            pred = labels[j]
            count = cm[i][j]
            if i == j:
                row.append(f"True Positives: {count} {actual} correctly classified as {pred}")
            else:
                row.append(f"Misclassified: {count} actual {actual} predicted as {pred}")
        hover_text.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 16, "family": "Outfit, sans-serif"},
        hoverinfo='text',
        hovertext=hover_text,
        showscale=True
    ))
    
    fig.update_layout(
        title="Confusion Matrix Heatmap",
        xaxis=dict(title="Predicted Species", title_font=dict(color="#B2BEC3")),
        yaxis=dict(title="Actual Species", title_font=dict(color="#B2BEC3")),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif")
    )
    return fig

def plot_knn_space(X_train, y_train, query_point, k, feature_indices, feature_names, class_names, scaler):
    """
    Plots the interactive nearest-neighbor visualization.
    Projects raw features into the selected 2D space, fits a NearestNeighbors classifier in scaled space,
    finds the K nearest neighbors of the query point, and highlights them in the visualization.
    """
    x_idx, y_idx = feature_indices
    x_name = feature_names[x_idx]
    y_name = feature_names[y_idx]
    
    # Scale query point and train set using fitted scaler to find actual neighbors in scaled space
    X_train_scaled = scaler.transform_test(X_train) # Safe scaling
    query_point_scaled = scaler.transform_input([query_point])
    
    # Fit NearestNeighbors in 4D space
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(query_point_scaled)
    neighbor_indices = indices[0]
    
    # Convert data to DataFrame for Plotly
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df['Species'] = [class_names[t].capitalize() for t in y_train]
    train_df['IsNeighbor'] = 'Normal Training Point'
    train_df.loc[neighbor_indices, 'IsNeighbor'] = 'Nearest Neighbor'
    
    fig = go.Figure()
    
    # Plot normal training points
    for target_val, color in CLASS_COLORS.items():
        species_label = class_names[target_val].capitalize()
        subset = train_df[(y_train == target_val) & (train_df['IsNeighbor'] == 'Normal Training Point')]
        
        fig.add_trace(go.Scatter(
            x=subset[x_name], y=subset[y_name],
            mode='markers',
            name=f"{species_label}",
            marker=dict(color=color, size=7, opacity=0.4),
            hovertext=[f"Species: {species_label}<br>Is Neighbor: No" for _ in range(len(subset))]
        ))
        
    # Plot nearest neighbors with highlighted styling
    for target_val, color in CLASS_COLORS.items():
        species_label = class_names[target_val].capitalize()
        subset = train_df[(y_train == target_val) & (train_df['IsNeighbor'] == 'Nearest Neighbor')]
        if len(subset) == 0:
            continue
            
        fig.add_trace(go.Scatter(
            x=subset[x_name], y=subset[y_name],
            mode='markers',
            name=f"Neighbor: {species_label}",
            marker=dict(
                color=color, size=12, symbol='circle',
                line=dict(color='#FFFFFF', width=2),
                opacity=1.0
            ),
            hovertext=[f"Species: {species_label}<br>Is Neighbor: YES (K-Nearest)" for _ in range(len(subset))]
        ))
        
    # Draw line connectors from query point to neighbors
    for idx in neighbor_indices:
        neighbor_x = train_df.loc[idx, x_name]
        neighbor_y = train_df.loc[idx, y_name]
        fig.add_trace(go.Scatter(
            x=[query_point[x_idx], neighbor_x],
            y=[query_point[y_idx], neighbor_y],
            mode='lines',
            line=dict(color='#FFEAA7', width=1, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
    # Plot the query point itself
    fig.add_trace(go.Scatter(
        x=[query_point[x_idx]], y=[query_point[y_idx]],
        mode='markers',
        name='Unseen Query Point',
        marker=dict(color='#FDCB6E', size=16, symbol='star', line=dict(color='#FFFFFF', width=2)),
        hovertext=[f"Query Point<br>{x_name}: {query_point[x_idx]}<br>{y_name}: {query_point[y_idx]}"],
        hoverinfo='text'
    ))
    
    # Update layout details
    fig.update_layout(
        title=f"K-Nearest Neighbors Lookup (K={k}) Projected on {x_name} vs {y_name}",
        xaxis=dict(title=x_name, gridcolor="#2D3436"),
        yaxis=dict(title=y_name, gridcolor="#2D3436"),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Outfit, sans-serif"),
        legend=dict(bgcolor='rgba(45, 52, 54, 0.5)')
    )
    
    neighbor_species = [class_names[y_train[i]].capitalize() for i in neighbor_indices]
    vote_counts = {c.capitalize(): neighbor_species.count(c.capitalize()) for c in class_names}
    neighbor_details = {
        'indices': neighbor_indices,
        'distances': distances[0],
        'species': neighbor_species,
        'vote_counts': vote_counts
    }
    return fig, neighbor_details

def get_nearest_neighbors_info(X_train, y_train, query_point, k, feature_names, class_names, scaler):
    """
    Computes and formats detailed information about the K nearest training samples to a query point.
    """
    X_train_scaled = scaler.transform_test(X_train)
    query_point_scaled = scaler.transform_input(query_point)
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(X_train_scaled)
    distances, indices = nn.kneighbors(query_point_scaled)
    
    records = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        row = {
            'Rank': f"#{rank}",
            'Training Sample ID': f"Sample #{idx}",
            'Species': class_names[y_train[idx]].capitalize(),
            'Scaled Distance': round(float(dist), 4)
        }
        for f_idx, f_name in enumerate(feature_names):
            row[f_name] = round(float(X_train[idx][f_idx]), 2)
        records.append(row)
        
    return pd.DataFrame(records)

