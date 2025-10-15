"""
Real-Time Monitoring Component
Live progress tracking and statistics display
"""

import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import plotly.graph_objects as go
from utils.api_client import APIClient


def render_monitor_section(api_base_url: str):
    """
    Render real-time monitoring interface.
    
    Args:
        api_base_url: Base URL of FastAPI backend
    """
    
    api_client = APIClient(api_base_url)
    
    st.subheader("🎥 Real-Time Monitoring")
    
    # Check if there's a job to monitor
    if 'current_job_id' not in st.session_state:
        st.info("👈 Upload a video in the **Upload** tab to start monitoring")
        st.markdown("---")
        st.markdown("### What You'll See Here:")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📊 Live Statistics")
            st.write("- Frames processed")
            st.write("- Vehicles tracked")
            st.write("- Plates detected")
            st.write("- Speed violations")
        
        with col2:
            st.markdown("#### ⚡ Performance Metrics")
            st.write("- Processing speed (FPS)")
            st.write("- Time elapsed")
            st.write("- Time remaining")
            st.write("- Progress percentage")
        
        with col3:
            st.markdown("#### 📈 Live Charts")
            st.write("- Detection trend")
            st.write("- Processing speed")
            st.write("- Violation timeline")
            st.write("- Track count over time")
        
        return
    
    # Get current job info
    job_id = st.session_state['current_job_id']
    filename = st.session_state.get('current_job_filename', 'Unknown')
    
    # Job header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### 📹 {filename}")
        st.caption(f"Job ID: `{job_id}`")
    
    with col2:
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        auto_refresh = st.toggle("Auto-refresh", value=True)
    
    st.divider()
    
    # Fetch job status
    with st.spinner("Fetching job status..."):
        status_data = api_client.get_job_status(job_id)
    
    if not status_data:
        st.error("❌ Could not fetch job status. Job might not exist.")
        if st.button("Clear Current Job"):
            del st.session_state['current_job_id']
            if 'current_job_filename' in st.session_state:
                del st.session_state['current_job_filename']
            st.rerun()
        return
    
    # Display status
    display_job_status(status_data)

    if status_data and status_data['status'] == 'processing':
        st.markdown("---")
        from components.live_video import render_live_video_section
        render_live_video_section(job_id, api_base_url.replace('http', 'ws'))
        st.markdown("---")
    
    # Display progress
    display_progress(status_data)
    
    # Display statistics
    if status_data.get('stats'):
        display_statistics(status_data['stats'], status_data)
    
    # Display processing metrics
    display_processing_metrics(status_data)
    
    # Display charts
    if status_data.get('stats'):
        display_charts(status_data)
    
    # Auto-refresh logic
    if auto_refresh and status_data['status'] == 'processing':
        # Store start time if not exists
        if 'monitor_start_time' not in st.session_state:
            st.session_state['monitor_start_time'] = time.time()
        
        # Refresh every 2 seconds
        time.sleep(2)
        st.rerun()
    elif status_data['status'] in ['completed', 'failed']:
        # Clear auto-refresh when done
        if 'monitor_start_time' in st.session_state:
            del st.session_state['monitor_start_time']


def display_job_status(status_data: Dict[str, Any]):
    """Display job status with visual indicators."""
    
    status = status_data['status']
    
    status_config = {
        'queued': ('⏳', 'info', 'Queued - Waiting to start'),
        'processing': ('🔄', 'info', 'Processing - In progress'),
        'completed': ('✅', 'success', 'Completed - Successfully processed'),
        'failed': ('❌', 'error', 'Failed - Error occurred')
    }
    
    icon, status_type, message = status_config.get(
        status, 
        ('⚠️', 'warning', f'Unknown status: {status}')
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if status_type == 'success':
            st.success(f"{icon} {message}")
        elif status_type == 'error':
            st.error(f"{icon} {message}")
            if 'error' in status_data:
                st.error(f"**Error Details:** {status_data['error']}")
        elif status_type == 'warning':
            st.warning(f"{icon} {message}")
        else:
            st.info(f"{icon} {message}")


def display_progress(status_data: Dict[str, Any]):
    """Display progress bar and completion info."""
    
    st.markdown("### 📊 Progress")
    
    progress = status_data.get('progress', 0)
    
    # Progress bar
    st.progress(progress / 100.0)
    
    # Progress details
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Progress",
            f"{progress:.1f}%",
            delta=None
        )
    
    # Calculate time estimates if processing
    if status_data['status'] == 'processing' and 'stats' in status_data:
        stats = status_data['stats']
        processed = stats.get('processed_frames', 0)
        total = stats.get('total_frames', 1)
        
        if processed > 0:
            # Estimate time remaining
            if 'monitor_start_time' in st.session_state:
                elapsed = time.time() - st.session_state['monitor_start_time']
                frames_per_sec = processed / elapsed if elapsed > 0 else 0
                remaining_frames = total - processed
                
                if frames_per_sec > 0:
                    eta_seconds = remaining_frames / frames_per_sec
                    eta = str(timedelta(seconds=int(eta_seconds)))
                    
                    with col2:
                        st.metric("Time Elapsed", str(timedelta(seconds=int(elapsed))))
                    
                    with col3:
                        st.metric("ETA", eta)
                else:
                    with col2:
                        st.metric("Time Elapsed", "Calculating...")
                    with col3:
                        st.metric("ETA", "Calculating...")
            else:
                with col2:
                    st.metric("Frames", f"{processed:,} / {total:,}")


def display_statistics(stats: Dict[str, Any], status_data: Dict[str, Any]):
    """Display live processing statistics."""
    
    st.markdown("### 📈 Live Statistics")
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    processed_frames = stats.get('processed_frames', 0)
    total_frames = stats.get('total_frames', 0)
    total_detections = stats.get('total_detections', 0)
    total_tracks = stats.get('total_tracks', 0)
    plate_detections = stats.get('plate_detections', 0)
    screenshots_saved = stats.get('screenshots_saved', 0)
    violations_count = stats.get('violations_count', 0)
    
    with col1:
        st.metric(
            "🎬 Frames Processed",
            f"{processed_frames:,}",
            delta=f"of {total_frames:,}" if total_frames > 0 else None
        )
        
        # Calculate detection rate
        if processed_frames > 0:
            detection_rate = (total_detections / processed_frames)
            st.caption(f"Avg: {detection_rate:.1f} detections/frame")
    
    with col2:
        st.metric(
            "🚗 Vehicles Tracked",
            f"{total_tracks:,}",
            delta=f"{total_detections:,} total detections" if total_detections > 0 else None
        )
        
        # Tracking efficiency
        if total_detections > 0:
            efficiency = (total_tracks / total_detections) * 100
            st.caption(f"Tracking: {efficiency:.0f}%")
    
    with col3:
        st.metric(
            "🔖 Plates Detected",
            f"{plate_detections:,}",
            delta=f"{screenshots_saved:,} saved" if screenshots_saved > 0 else None
        )
        
        # Quality rate
        if plate_detections > 0:
            quality_rate = (screenshots_saved / plate_detections) * 100
            st.caption(f"Quality: {quality_rate:.0f}%")
    
    with col4:
        # Violations with color coding
        if violations_count > 0:
            st.metric(
                "⚠️ Speed Violations",
                f"{violations_count:,}",
                delta="Action needed",
                delta_color="inverse"
            )
        else:
            st.metric(
                "✅ Speed Violations",
                "0",
                delta="All clear",
                delta_color="normal"
            )


def display_processing_metrics(status_data: Dict[str, Any]):
    """Display processing performance metrics."""
    
    if status_data['status'] != 'processing':
        return
    
    if 'stats' not in status_data:
        return
    
    st.markdown("### ⚡ Performance Metrics")
    
    stats = status_data['stats']
    processed = stats.get('processed_frames', 0)
    
    if 'monitor_start_time' in st.session_state and processed > 0:
        elapsed = time.time() - st.session_state['monitor_start_time']
        
        if elapsed > 0:
            fps = processed / elapsed
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Processing Speed",
                    f"{fps:.1f} FPS",
                    delta="Frames per second"
                )
            
            with col2:
                frames_per_minute = fps * 60
                st.metric(
                    "Throughput",
                    f"{frames_per_minute:.0f}",
                    delta="Frames/minute"
                )
            
            with col3:
                st.metric(
                    "Time Elapsed",
                    f"{int(elapsed)}s",
                    delta=f"{elapsed/60:.1f} min"
                )
            
            with col4:
                # Efficiency indicator
                if fps >= 20:
                    efficiency = "🟢 Excellent"
                elif fps >= 10:
                    efficiency = "🟡 Good"
                else:
                    efficiency = "🔴 Slow"
                
                st.metric(
                    "Efficiency",
                    efficiency,
                    delta=None
                )


def display_charts(status_data: Dict[str, Any]):
    """Display live charts and visualizations."""
    
    st.markdown("### 📊 Analytics")
    
    stats = status_data.get('stats', {})
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Detection breakdown pie chart
        fig_pie = create_detection_breakdown_chart(stats)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Processing progress gauge
        fig_gauge = create_progress_gauge(status_data.get('progress', 0))
        st.plotly_chart(fig_gauge, use_container_width=True)


def create_detection_breakdown_chart(stats: Dict[str, Any]) -> go.Figure:
    """Create pie chart showing detection breakdown."""
    
    total_detections = stats.get('total_detections', 0)
    total_tracks = stats.get('total_tracks', 0)
    plate_detections = stats.get('plate_detections', 0)
    
    # Calculate untracked detections
    untracked = max(0, total_detections - total_tracks)
    
    labels = ['Tracked Vehicles', 'Untracked Detections', 'Plates Detected']
    values = [total_tracks, untracked, plate_detections]
    colors = ['#00CC96', '#AB63FA', '#FFA15A']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        hole=0.3
    )])
    
    fig.update_layout(
        title="Detection Breakdown",
        showlegend=True,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def create_progress_gauge(progress: float) -> go.Figure:
    """Create gauge chart for progress visualization."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=progress,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Processing Progress"},
        delta={'reference': 100, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightblue"},
                {'range': [75, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig