# Check Employee Attendance System

A face recognition system for employee attendace tracking using FAISS vector search and deep learning embeddings for real-time indentification.

## Live Demo

> **Try the live demo above to test system without any setup.**

## Technology Stack

- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu==1.11.0`)
- **Web Framework**: Streamlit (`streamlit==1.46.1`)
- **Deep Learning**: InceptionResNetV1 via `facenet-pytorch==2.6.0`, `torchvision==0.17.2`
- **Data Processing**: NumPy (`numpy==1.26.0`), OpenCV (`opencv-python-headless==4.8.1.78`), Pandas (`pandas==2.3.1`)
- **Visualization**: Matplotlib (`matplotlib==3.10.3`), Plotly (`plotly==6.2.0`)
- **Machine Learning**: scikit-learn (`scikit-learn==1.7.0`)
- **Language**: Python 3.11

## Local Setup

```bash
# Clone the repository
git clone https://github.com/nguyenvang12003454/Face-Based-Employee-Check-in-System.git
cd Face-Based-Employee-Check-in-System

# Install dependencies
pip install -r requirements.txt

# Add your tailord images in ./Dataset folder

# Start the Streamlit app   
streamlit run app.py
```

## Project Structure

```text
Face-Based-Employee-Check-in-System/
├── app.py                  # Main Streamlit application
├── facenet_label_map.npy   # Labels of Index
├── facenet_features.index  # Faiss Index
├── requirements.txt        # Python dependencies
├── Dataset/Dataset         # Dataset
│   └── Avatar_employee-name.jpg
└── README.md           # This file
```