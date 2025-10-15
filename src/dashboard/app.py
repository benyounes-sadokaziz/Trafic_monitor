"""
Traffic Monitor Dashboard
Main Streamlit application for real-time video processing monitoring
"""

import streamlit as st
from datetime import datetime

# Import components
from components.monitor import render_monitor_section
from components.upload import render_upload_section
from utils.api_client import APIClient

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Traffic Monitor Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Initialize API client
@st.cache_resource
def get_api_client():
    """Create and cache API client."""
    return APIClient(API_BASE_URL)


def main():
    """Main dashboard application."""
    
    # Get API client
    api_client = get_api_client()
    
    # Header
    st.title("🚗 Traffic Monitor Dashboard")
    st.markdown("Real-time vehicle detection, tracking, and speed monitoring")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Status")
        
        # Check API health
        health = api_client.health_check()
        
        if health:
            st.success("✅ API Online")
            with st.expander("API Details"):
                st.json(health)
        else:
            st.error("❌ API Offline")
            st.warning("Please start FastAPI backend:")
            st.code("uvicorn src.api.main:app --reload", language="bash")
        
        st.divider()
        
        st.header("📊 Quick Stats")
        
        # Get job statistics
        jobs_data = api_client.list_all_jobs()
        if jobs_data:
            total_jobs = jobs_data.get('total', 0)
            
            # Count jobs by status
            jobs = jobs_data.get('jobs', [])
            completed = sum(1 for j in jobs if j.get('status') == 'completed')
            processing = sum(1 for j in jobs if j.get('status') == 'processing')
            failed = sum(1 for j in jobs if j.get('status') == 'failed')
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Jobs", total_jobs)
                st.metric("✅ Completed", completed)
            with col2:
                st.metric("⏳ Processing", processing)
                st.metric("❌ Failed", failed)
        else:
            st.metric("Active Jobs", "N/A")
            st.metric("Total Processed", "N/A")
        
        # Auto-refresh toggle
        st.divider()
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)
        
        if auto_refresh:
            st.info("Dashboard will refresh every 5 seconds")
    
    # Main content area
    st.header("🎬 Video Processing")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "🎥 Monitor", "📋 History"])
    
    with tab1:
        # Render upload section from component
        render_upload_section(API_BASE_URL)
    
    with tab2:
        # Import monitor component
        
        
        # Render monitor section
        render_monitor_section(API_BASE_URL)
    
    with tab3:
        st.subheader("Job History")
        
        # Fetch all jobs
        jobs_data = api_client.list_all_jobs()
        
        if jobs_data and jobs_data.get('jobs'):
            st.write(f"**Total Jobs:** {jobs_data['total']}")
            
            # Display jobs as cards
            for job in jobs_data['jobs'][:10]:  # Show last 10 jobs
                with st.container():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{job['filename']}**")
                        st.caption(f"ID: {job['job_id'][:8]}...")
                    
                    with col2:
                        status = job['status']
                        if status == 'completed':
                            st.success("✅ Done")
                        elif status == 'processing':
                            st.info("⏳ Processing")
                        elif status == 'failed':
                            st.error("❌ Failed")
                        else:
                            st.warning(f"⚠️ {status}")
                    
                    with col3:
                        progress = job.get('progress', 0)
                        st.write(f"{progress:.0f}%")
                    
                    with col4:
                        if st.button("View", key=f"view_{job['job_id']}"):
                            st.session_state['current_job_id'] = job['job_id']
                            st.session_state['current_job_filename'] = job['filename']
                            st.info("👉 Switch to Monitor tab")
                    
                    st.divider()
        else:
            st.info("No jobs found. Upload a video to get started!")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.caption(f"Traffic Monitor v1.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
    with col3:
        if health:
            st.caption("🟢 System Online")
        else:
            st.caption("🔴 System Offline")
    
    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()