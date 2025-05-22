import streamlit as st
import pandas as pd
import numpy as np
import io
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import required libraries for audio and text processing
try:
    import whisper
    import torch
    import torchaudio
    import librosa
    from transformers import pipeline
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Audio & Text Sentiment Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Login credentials
VALID_USERNAME = "media@firsteconomy.com"
VALID_PASSWORD = "fe@1234"

def check_login():
    """Check if user is logged in"""
    return st.session_state.get('logged_in', False)

def login_page():
    """Display login page"""
    st.title("🔐 Login to Sentiment Analysis App")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Please enter your credentials")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Login", use_container_width=True):
                if username == VALID_USERNAME and password == VALID_PASSWORD:
                    st.session_state['logged_in'] = True
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            st.markdown("---")
            st.info("**Demo Credentials:**\n\nUsername: media@firsteconomy.com\n\nPassword: fe@1234")

@st.cache_resource
def load_whisper_model():
    """Load Whisper model with caching"""
    return whisper.load_model("small")

@st.cache_resource
def load_sentiment_models():
    """Load sentiment analysis models with caching"""
    try:
        sentiment_pipe = pipeline("sentiment-analysis", 
                                model="nlptown/bert-base-multilingual-uncased-sentiment")
        classifier = pipeline("zero-shot-classification", 
                            model="facebook/bart-large-mnli")
        text_classifier = pipeline("zero-shot-classification", 
                                 model="joeddav/xlm-roberta-large-xnli")
        return sentiment_pipe, classifier, text_classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def transcribe_audio(audio_file, model):
    """Transcribe audio file using Whisper"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # Transcribe audio
        result = model.transcribe(tmp_file_path)
        transcription = result["text"]
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return transcription
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

def analyze_audio_sentiment(transcription, sentiment_pipe, classifier):
    """Analyze sentiment of transcribed audio"""
    try:
        # Basic sentiment (1-5 stars)
        sentiment_result = sentiment_pipe(transcription)
        
        # Advanced tone classification
        labels = [
            "rude", "polite", "friendly", "angry", "sarcastic", "casual", "formal",
            "aggressive", "respectful", "smooth", "awkward", "relaxed", "excited", "tense"
        ]
        
        classification = classifier(transcription, candidate_labels=labels, multi_label=True)
        
        # Sort by score descending
        top_labels = list(zip(classification['labels'], classification['scores']))
        top_labels.sort(key=lambda x: x[1], reverse=True)
        
        return sentiment_result, top_labels
    except Exception as e:
        st.error(f"Error analyzing sentiment: {e}")
        return None, None

def create_audio_charts(sentiment_result, top_labels):
    """Create charts for audio sentiment analysis"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Overall Sentiment', 'Top 5 Tone Characteristics', 
                       'Confidence Score', 'All Tone Characteristics'),
        specs=[[{"type": "indicator"}, {"type": "bar"}],
               [{"type": "gauge"}, {"type": "bar"}]]
    )
    
    # Overall sentiment gauge
    sentiment_score = float(sentiment_result[0]['label'].split()[0])
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=sentiment_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment (1-5 Stars)"},
            gauge={
                'axis': {'range': [None, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "lightgray"},
                    {'range': [2, 4], 'color': "gray"},
                    {'range': [4, 5], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 4.5
                }
            }
        ),
        row=1, col=1
    )
    
    # Top 5 characteristics bar chart
    top_5 = top_labels[:5]
    fig.add_trace(
        go.Bar(
            x=[score*100 for _, score in top_5],
            y=[label.capitalize() for label, _ in top_5],
            orientation='h',
            marker_color='lightblue'
        ),
        row=1, col=2
    )
    
    # Confidence score
    confidence = sentiment_result[0]['score'] * 100
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=confidence,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Confidence %"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ]
            }
        ),
        row=2, col=1
    )
    
    # All characteristics
    all_labels = top_labels[:10]  # Show top 10
    fig.add_trace(
        go.Bar(
            x=[label.capitalize() for label, _ in all_labels],
            y=[score*100 for _, score in all_labels],
            marker_color='orange'
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Confidence %", row=1, col=2)
    fig.update_yaxes(title_text="Characteristics", row=1, col=2)
    fig.update_xaxes(title_text="Characteristics", row=2, col=2)
    fig.update_yaxes(title_text="Confidence %", row=2, col=2)
    
    return fig

def analyze_text_sentiment(df, text_classifier):
    """Analyze sentiment of text data"""
    try:
        # Combine remarks and group by Campaign
        df["Combined_Remark"] = df["Remark"].astype(str) + " " + df["FE Remark"].astype(str)
        grouped_df = df.groupby("Campaign/ Ad Set")["Combined_Remark"].apply(
            lambda x: " || ".join(x)
        ).reset_index()
        
        # Define categories
        labels = {
            "Interest": ["interested", "not interested", "follow-up", "disqualified", "doubtful"],
            "Attitude": ["polite", "rude", "confused", "serious", "casual", "formal", "awkward", "smooth"],
            "Behavior": ["spam", "genuine", "wrong number", "number switched off", "looking for rental"],
            "Engagement": ["unreachable", "connected", "planning site visit", "not responding"]
        }
        
        def classify_all(text):
            result = {}
            for key, lbls in labels.items():
                output = text_classifier(text, lbls, multi_label=True)
                top_labels = sorted(
                    zip(output["labels"], output["scores"]), 
                    key=lambda x: x[1], reverse=True
                )[:2]
                result[key] = ", ".join([f"{l} ({round(s*100)}%)" for l, s in top_labels])
            return pd.Series(result)
        
        # Run sentiment analysis
        sentiment_results = grouped_df["Combined_Remark"].apply(classify_all)
        
        # Merge results
        final_df = pd.concat([grouped_df["Campaign/ Ad Set"], sentiment_results], axis=1)
        
        return final_df
    except Exception as e:
        st.error(f"Error analyzing text sentiment: {e}")
        return None

def main_app():
    """Main application interface"""
    st.title("🎤📊 Audio & Text Sentiment Analysis Dashboard")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Audio Analysis", "Text Analysis", "About"]
    )
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    # Load models
    with st.spinner("Loading AI models..."):
        sentiment_pipe, classifier, text_classifier = load_sentiment_models()
        
        if None in [sentiment_pipe, classifier, text_classifier]:
            st.error("Failed to load models. Please check your internet connection and try again.")
            return
    
    if analysis_type == "Audio Analysis":
        st.header("🎤 Audio Sentiment Analysis")
        
        uploaded_audio = st.file_uploader(
            "Upload an audio file (.wav, .mp3, .m4a)", 
            type=['wav', 'mp3', 'm4a']
        )
        
        if uploaded_audio is not None:
            st.audio(uploaded_audio, format='audio/wav')
            
            if st.button("Analyze Audio", type="primary"):
                with st.spinner("Loading Whisper model..."):
                    whisper_model = load_whisper_model()
                
                with st.spinner("Transcribing audio..."):
                    transcription = transcribe_audio(uploaded_audio, whisper_model)
                
                if transcription:
                    st.success("Transcription completed!")
                    
                    # Display transcription
                    st.subheader("📝 Transcription")
                    st.write(transcription)
                    
                    with st.spinner("Analyzing sentiment..."):
                        sentiment_result, top_labels = analyze_audio_sentiment(
                            transcription, sentiment_pipe, classifier
                        )
                    
                    if sentiment_result and top_labels:
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("📊 Basic Sentiment")
                            st.json(sentiment_result[0])
                        
                        with col2:
                            st.subheader("🎭 Top Tone Characteristics")
                            for label, score in top_labels[:5]:
                                st.write(f"**{label.capitalize()}:** {round(score*100, 1)}%")
                        
                        # Create and display charts
                        st.subheader("📈 Detailed Analysis Charts")
                        fig = create_audio_charts(sentiment_result, top_labels)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download results
                        results_df = pd.DataFrame({
                            'Characteristic': [label for label, _ in top_labels],
                            'Confidence_Score': [f"{round(score*100, 1)}%" for _, score in top_labels]
                        })
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="audio_sentiment_analysis.csv",
                            mime="text/csv"
                        )
    
    elif analysis_type == "Text Analysis":
        st.header("📊 Text Sentiment Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a file (.csv or .xlsx)", 
            type=['csv', 'xlsx']
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = df.fillna("")
                
                st.success(f"File loaded successfully! Shape: {df.shape}")
                
                # Display sample data
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Check required columns
                required_columns = ["Campaign/ Ad Set", "Remark", "FE Remark"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                    st.info("Required columns: Campaign/ Ad Set, Remark, FE Remark")
                else:
                    if st.button("Analyze Text Sentiment", type="primary"):
                        with st.spinner("Running sentiment analysis..."):
                            final_df = analyze_text_sentiment(df, text_classifier)
                        
                        if final_df is not None:
                            st.success("Analysis completed!")
                            
                            # Display results
                            st.subheader("📊 Sentiment Analysis Results")
                            st.dataframe(final_df)
                            
                            # Create summary charts
                            st.subheader("📈 Summary Charts")
                            
                            # Extract interest data for visualization
                            interest_data = []
                            for idx, row in final_df.iterrows():
                                interest_str = row['Interest']
                                # Parse the interest string to extract main sentiment
                                if 'interested' in interest_str.lower():
                                    if 'not interested' in interest_str.lower():
                                        interest_data.append('Not Interested')
                                    else:
                                        interest_data.append('Interested')
                                elif 'follow-up' in interest_str.lower():
                                    interest_data.append('Follow-up')
                                elif 'doubtful' in interest_str.lower():
                                    interest_data.append('Doubtful')
                                else:
                                    interest_data.append('Other')
                            
                            # Create interest distribution chart
                            interest_counts = pd.Series(interest_data).value_counts()
                            fig_interest = px.pie(
                                values=interest_counts.values,
                                names=interest_counts.index,
                                title="Interest Level Distribution"
                            )
                            st.plotly_chart(fig_interest, use_container_width=True)
                            
                            # Download results
                            csv = final_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="text_sentiment_analysis.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    elif analysis_type == "About":
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## 🎯 Purpose
        This application provides comprehensive sentiment analysis for both audio and text data.
        
        ## 🎤 Audio Analysis Features
        - **Speech-to-Text:** Uses OpenAI's Whisper model for accurate transcription
        - **Sentiment Analysis:** 1-5 star rating system
        - **Tone Detection:** Identifies emotional characteristics (polite, rude, friendly, etc.)
        - **Visual Analytics:** Interactive charts and gauges
        
        ## 📊 Text Analysis Features
        - **Campaign Analysis:** Groups feedback by campaign/ad set
        - **Multi-category Classification:** Interest, Attitude, Behavior, Engagement
        - **Batch Processing:** Handles CSV and Excel files
        - **Summary Visualizations:** Distribution charts
        
        ## 🛠️ Technical Stack
        - **Streamlit:** Web application framework
        - **Whisper:** Audio transcription
        - **Transformers:** Sentiment analysis models
        - **Plotly:** Interactive visualizations
        - **Pandas:** Data processing
        
        ## 📝 Supported Formats
        - **Audio:** .wav, .mp3, .m4a
        - **Text:** .csv, .xlsx
        
        ## 🔒 Security
        - Secure login system
        - Session management
        - No data persistence
        """)

def main():
    """Main function to run the app"""
    if not check_login():
        login_page()
    else:
        main_app()

if __name__ == "__main__":
    main()
