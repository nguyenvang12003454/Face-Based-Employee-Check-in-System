import os
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import faiss
import torch
import torchvision.transforms as T
import time
from facenet_pytorch import InceptionResnetV1
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
import base64
from io import BytesIO

# Page Configuration
st.set_page_config(page_title="Employee Dashboard", page_icon="🧑‍💼", layout="wide")
st.markdown("""
    <style>
        .employee-card {
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }
        .employee-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .match-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px 0;
            gap: 20px;
        }
        .vector-space {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .top-matches {
            display: grid;
            grid-template-columns: repeat(2, 1fr); /* Two columns */
            gap: 15px;
            padding: 10px 0;
        }

        .match-item {
            text-align: center;
        }
        .status-badge {
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.8em;
        }
        .checked-in {
            background-color: #d4edda;
            color: #155724;
            margin: 0 0 20px 0;
        }
        .not-checked {
            background-color: #f8d7da;
            color: #721c24;
            margin: 0 0 20px 0;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>🧑‍💼 Face-Based Employee Check-in System</h1>", unsafe_allow_html=True)

# Functional code
@st.cache_resource
def load_facenet_model():
    """Load the pre-trained InceptionResnetV1 model"""
    return InceptionResnetV1(pretrained='vggface2').eval()

@st.cache_resource
def load_faiss_index():
    """Load the FAISS index and label map"""
    if not os.path.exists("facenet_features.index") or not os.path.exists("facenet_label_map.npy"):
        st.error("FAISS index or label map not found.")
        return None, None, None
    
    index = faiss.read_index("facenet_features.index")
    label_map = np.load("facenet_label_map.npy")

    embeddings = index.reconstruct_n(0, index.ntotal)

    return index, label_map, embeddings

facenet_recognition_model = load_facenet_model()
index, label_map, all_embeddings = load_faiss_index()

transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def crop_center_square(image):
    """Crop the center square of an image and resize to 300*300"""
    width, height = image.size
    size = min(width, height)
    left = (width - size)/2
    top = (height - size)/2
    right = (width + size)/2
    bottom = (height + size)/2

    image = image.crop((left, top, right, bottom))
    return image

def image_to_feature(image, model):
    img = image.convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze().numpy()

def search_similar_features(query_feature, k=5, threshold=0.3):
    if index is None or label_map is None:
        return []
    
    # Search the index
    similarities, indices = index.search(np.array([query_feature]), k)

    results = []
    for i in range(len(indices[0])):
        similarity = similarities[0][i]

        if similarity > threshold:
            employee_name = label_map[indices[0][i]]
            results.append((employee_name, similarity, indices[0][i]))
    return results

def visualize_embeddings(query_embedding=None, matches=None):
    """Visualize embeddings in 3D space usnig t-SNE with highlighted points and similarity lines"""
    if all_embeddings is None or len(all_embeddings) < 3:
        st.warning("Not enough embeddings to visualize (need at least 3).")
        return
    
    # Prepare data for visualization
    embeddings_to_plot = all_embeddings.copy()
    labels = label_map.copy()
    colors = ['blue']*len(all_embeddings)
    sizes = [8] * len(all_embeddings)
    # hover_names = [f"Employee: {name}" for name in labels]
    hover_names = []
    for i, name in enumerate(labels):
        if query_embedding is not None and i < len(all_embeddings):
            # Calculate similarity from this point to query
            dist = np.linalg.norm(all_embeddings[i] - query_embedding)
            hover_names.append(f"Employee: {name} (Dist: {dist:.4f})")
        else:
            hover_names.append(f"Employee: {name}")
    
    # Add query embedding if available
    query_idx = None
    if query_embedding is not None:
        query_idx = len(embeddings_to_plot)  # Store the index of the query point
        embeddings_to_plot = np.vstack([embeddings_to_plot, query_embedding])
        labels = np.append(labels, "Query Face")
        colors.append('red')
        sizes.append(12)
        hover_names.append("Your Face (Query)")
    
    # Highlight matches if available
    match_indices = []
    if matches:
        for i, (name, similarity, idx) in enumerate(matches):
            if idx < len(colors):  # Ensure we don't go out of bounds
                colors[idx] = 'green' if i == 0 else 'orange'  # Top match is green, others orange
                sizes[idx] = 12
                hover_names[idx] = f"Match {i+1}: {name} (Similarity: {similarity:.4f})"
                match_indices.append(idx)
    
    # Calculate appropriate perplexity (must be less than n_samples)
    n_samples = len(embeddings_to_plot)
    perplexity = min(30, max(5, n_samples - 1))  # Use 5-30 range, but always < n_samples
    
    # Reduce dimensionality with t-SNE (3D)
    # tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    # embeddings_3d = tsne.fit_transform(embeddings_to_plot)
    
    from sklearn.decomposition import PCA

    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(embeddings_to_plot)
    # Create a DataFrame for Plotly (now with z-axis)
    df = pd.DataFrame({
        'x': embeddings_3d[:, 0],
        'y': embeddings_3d[:, 1],
        'z': embeddings_3d[:, 2],
        'label': labels,
        'color': colors,
        'size': sizes,
        'hover_name': hover_names
    })
    
    # Create 3D interactive plot
    fig = px.scatter_3d(
        df, 
        x='x', 
        y='y', 
        z='z',
        color='color', 
        size='size',
        hover_name='hover_name', 
        title='Face Embeddings in 3D Space',
        color_discrete_map={
            'blue': 'rgba(30, 136, 229, 0.7)',  # Blue for employees
            'red': 'rgba(229, 57, 53, 1)',       # Red for query
            'green': 'rgba(67, 160, 71, 1)',     # Green for top match
            'orange': 'rgba(255, 152, 0, 1)'     # Orange for other matches
        }
    )
    
    # Add similarity lines between query and matches
    if query_embedding is not None and matches:
        for i, (name, similarity, idx) in enumerate(matches[:5]):  # Only show top 3 matches
            if idx < len(embeddings_3d):
                # Add a line between query and match
                fig.add_trace(
                    go.Scatter3d(
                        x=[embeddings_3d[query_idx, 0], embeddings_3d[idx, 0]],
                        y=[embeddings_3d[query_idx, 1], embeddings_3d[idx, 1]],
                        z=[embeddings_3d[query_idx, 2], embeddings_3d[idx, 2]],
                        mode='lines',
                        line=dict(
                            color='purple' if i == 0 else 'gray',
                            width=3 if i == 0 else 1
                        ),
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"Similarity: {similarity:.4f}"
                    )
                )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            zaxis_title='Dimension 3',
        ),
        showlegend=False,
        hovermode='closest',
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # Add annotations for matches
    if matches:
        for i, (name, similarity, idx) in enumerate(matches[:5]):  # Only annotate top 3
            if idx < len(embeddings_3d):
                fig.add_annotation(
                    x=embeddings_3d[idx, 0],
                    y=embeddings_3d[idx, 1],
                    text=f"Match {i+1}" if i > 0 else "Best Match",
                    showarrow=True,
                    arrowhead=1,
                    font=dict(size=12, color="black")
                )
    
    st.plotly_chart(fig, use_container_width=True)
    # Add a similarity table below the plot
    if matches:
        st.subheader("Match Distances")
        match_data = []
        for i, (name, similarity, idx) in enumerate(matches):
            match_data.append({
                "Rank": i+1,
                "Name": name,
                "Similarity": f"{similarity:.4f}",
            })
        
        df_matches = pd.DataFrame(match_data)
        st.dataframe(df_matches.style.highlight_max(subset=["Similarity"], color='lightgreen'), 
                    use_container_width=True)
    

def get_avatar_image(employee_name):
    """Get avatar image path for an employee."""
    avatar_path_jpg = f"./Dataset/Avatar_{employee_name}.jpg"
    avatar_path_JPG = f"./Dataset/Avatar_{employee_name}.JPG"
    
    if os.path.exists(avatar_path_jpg):
        return avatar_path_jpg
    elif os.path.exists(avatar_path_JPG):
        return avatar_path_JPG
    else:    
        return "https://via.placeholder.com/300?text=No+Photo"

# State management
if "checkin_status" not in st.session_state:
    if label_map is not None:
        st.session_state.checkin_status = {name: False for name in label_map}
    else:
        st.session_state.checkin_status = {}

if "captured_image" not in st.session_state:
    st.session_state.captured_image = None

if "matching_result" not in st.session_state:
    st.session_state.matching_result = None

if "matching_distance" not in st.session_state:
    st.session_state.matching_distance = None

if "matching_avatar" not in st.session_state:
    st.session_state.matching_avatar = None

if "query_embedding" not in st.session_state:
    st.session_state.query_embedding = None

if "all_matches" not in st.session_state:
    st.session_state.all_matches = []

# Main Layout
col1, col2 = st.columns([3, 2])

with col1:
    # Employee List Section
    st.markdown("### 👥 Employee List")
    
    if not st.session_state.checkin_status:
        st.warning("No employee data available. Please ensure the database is properly set up.")
    else:
        # Create a grid of employee cards
        cols = st.columns(3)
        for i, (name, checked) in enumerate(st.session_state.checkin_status.items()):
            with cols[i % 3]:
                avatar = get_avatar_image(name)
                print(avatar, i)
                status_class = "checked-in" if checked else "not-checked"
                status_text = "CHECKED IN" if checked else "NOT CHECKED"
                image = Image.open(avatar)
                
                image = image.resize((300, 300))
                
                st.image(image, use_container_width=True)

                st.markdown(f"""
                    <div style="text-align:center;">
                        <h4>{name}</h4>
                        <div class="status-badge {status_class}">{status_text}</div>
                    </div>
                """, unsafe_allow_html=True)

with col2:
    # Check-in Section
    st.markdown("### 📸 Employee Check-in")
    
    # Camera capture button
    if st.button("Open Camera for Check-in", use_container_width=True, type="primary"):
        st.session_state.capture_clicked = True
    
    if st.session_state.get('capture_clicked', False):
        st.info("Preparing camera...")
        st.session_state.captured_image = None
        camera_placeholder = st.empty()
        countdown_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        captured_frame = None

        if not cap.isOpened():
            st.warning("📷 Could not access camera. Please upload an image instead.")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                captured_image = Image.open(uploaded_file).convert("RGB")
                captured_image = crop_center_square(captured_image)
                st.session_state.captured_image = captured_image
        else:
            start_time = time.time()
            capture_delay = 3  # 3-second countdown

            while (time.time() - start_time) < capture_delay:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                remaining = int(capture_delay - (time.time() - start_time)) + 1
                countdown_placeholder.markdown(
                    f"<h2 style='text-align:center;color:#ff5722;'>{remaining}</h2>",
                    unsafe_allow_html=True
                )

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                if (time.time() - start_time) >= (capture_delay - 0.1):
                    captured_frame = frame_rgb

                time.sleep(0.05)

            cap.release()
            camera_placeholder.empty()
            countdown_placeholder.empty()

            if captured_frame is not None:
                captured_image = Image.fromarray(captured_frame)
                captured_image = crop_center_square(captured_image)
                st.session_state.captured_image = captured_image

        # If we have a captured image (from camera or upload)
        if st.session_state.get('captured_image'):
            with st.spinner("🔍 Finding match..."):
                query_embedding = image_to_feature(st.session_state.captured_image, face_recognition_model)
                st.session_state.query_embedding = query_embedding

                matches = search_similar_features(query_embedding, k=5, threshold=0.3)
                st.session_state.all_matches = matches

                if matches:
                    best_match_name, best_distance, _ = matches[0]
                    st.session_state.checkin_status[best_match_name] = True
                    st.session_state.matching_result = best_match_name
                    st.session_state.matching_distance = best_distance
                    st.session_state.matching_avatar = get_avatar_image(best_match_name)
                    st.session_state.capture_clicked = False
                    st.rerun()
                else:
                    st.session_state.matching_result = None

                
    
    # Show captured image if available
    if st.session_state.captured_image is not None:
        st.markdown("---")
        st.markdown("### 📷 Captured Image")
        st.image(st.session_state.captured_image, use_container_width=True)
        
        # Show matching results
        if st.session_state.matching_result:
            st.success(f"""
            ✅ **{st.session_state.matching_result}** checked in!
            Similarity: {st.session_state.matching_distance:.4f}
            """)
            
            def image_to_base64(image_path, size=(300, 300)):
                img = Image.open(image_path).resize(size)
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode()

            # Show top matches
            st.markdown("### 🏆 Top Matches")
            inner_cols = st.columns(2)  # Create two columns inside col2

            for i, (name, similarity, _) in enumerate(st.session_state.all_matches[:5]):
                avatar = get_avatar_image(name)
                border_color = "#4CAF50" if i == 0 else "#FFC107"

                with inner_cols[i % 2]:  # Alternate between the two inner columns
                    st.markdown(f"""
                        <div style="border: 3px solid {border_color}; border-radius: 8px; padding: 5px; margin-bottom:10px; text-align: center;">
                            <img src="data:image/png;base64,{image_to_base64(avatar)}" style="width:100%; border-radius:6px;" />
                            <div style="margin-top:5px;">
                                <strong>{name}</strong><br>
                                <small>Similarity: {similarity:.4f}</small>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Show vector space visualization
            st.markdown("### 🌌 Vector Space Visualization")
            st.markdown("""
            Each point represents a face embedding. The red point is your face, 
            green is the best match, and orange are other potential matches.
            """)
            visualize_embeddings(
                st.session_state.query_embedding,
                [(name, dist, idx) for name, dist, idx in st.session_state.all_matches[:5]]
            )
        elif st.session_state.matching_result is None and st.session_state.all_matches:
            st.error("No matches found above the confidence threshold.")
            visualize_embeddings(st.session_state.query_embedding)