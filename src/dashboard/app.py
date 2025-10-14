"""
Traffic Monitor Dashboard
Main Streamlit application for real-time video processing monitoring
"""

import streamlit as st
import requests
from datetime import datetime

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic Monitor Dashboard",
    page_icon="🚗",
    layout="wide",  # Use full width
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.json()
    except requests.exceptions.RequestException:
        return None


def main():
    """Main dashboard application."""
    
    # Header
    st.title("🚗 Traffic Monitor Dashboard")
    st.markdown("Real-time vehicle detection, tracking, and speed monitoring")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        health = check_api_health()
        
        if health:
            st.success("✅ API Online")
            st.json(health)
        else:
            st.error("❌ API Offline")
            st.warning("Please start FastAPI backend:")
            st.code("uvicorn src.api.main:app --reload")
        
        st.divider()
        
        st.header("📊 Quick Stats")
        # Placeholder for now
        st.metric("Active Jobs", "0")
        st.metric("Total Processed", "0")
        st.metric("Violations Today", "0")
    
    # Main content area
    st.header("🎬 Video Processing")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🎥 Monitor", "📋 History"])
    
    with tab1:
        st.subheader("Upload Video for Processing")
        st.info("Upload section coming in Step 2!")
        st.write("You'll be able to:")
        st.write("- Upload video files")
        st.write("- Set processing parameters")
        st.write("- Start processing jobs")
    
    with tab2:
        st.subheader("Real-Time Monitoring")
        st.info("Monitoring section coming in Step 3!")
        st.write("You'll see:")
        st.write("- Live video frames with bounding boxes")
        st.write("- Real-time progress updates")
        st.write("- Vehicle tracking statistics")
    
    with tab3:
        st.subheader("Job History")
        st.info("History section coming in Step 6!")
        st.write("You'll be able to:")
        st.write("- View all processed videos")
        st.write("- Download results")
        st.write("- Retry failed jobs")
    
    # Footer
    st.divider()
    st.caption(f"Traffic Monitor v1.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()