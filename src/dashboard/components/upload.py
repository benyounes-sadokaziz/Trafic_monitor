"""
Video Upload Component
Handles video file upload and job submission to API
"""

import streamlit as st
import requests
from pathlib import Path


def render_upload_section(api_base_url: str):
    """
    Render the video upload interface.
    
    Args:
        api_base_url: Base URL of FastAPI backend
    """
    
    st.subheader("📤 Upload Video for Processing")
    
    # Create two columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov'],
            help="Supported formats: MP4, AVI, MOV"
        )
        
        if uploaded_file is not None:
            # Show video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "File type": uploaded_file.type
            }
            
            st.success("✅ Video file loaded successfully!")
            
            # Display file info in a nice box
            with st.expander("📄 File Details", expanded=True):
                for key, value in file_details.items():
                    st.write(f"**{key}:** {value}")
            
            # Show video preview
            st.video(uploaded_file)
    
    with col2:
        st.markdown("### ⚙️ Processing Parameters")
        
        # Camera calibration parameters
        frame_width = st.number_input(
            "Frame Width (meters)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            help="Real-world width that the camera sees"
        )
        
        frame_height = st.number_input(
            "Frame Height (meters)",
            min_value=1.0,
            max_value=1000.0,
            value=30.0,
            step=1.0,
            help="Real-world height that the camera sees"
        )
        
        # Optional: Max frames to process
        process_all = st.checkbox("Process entire video", value=True)
        
        if not process_all:
            max_frames = st.number_input(
                "Max frames to process",
                min_value=10,
                max_value=10000,
                value=300,
                step=10,
                help="Limit processing for testing"
            )
        else:
            max_frames = None
        
        # Save output video option
        save_output = st.checkbox(
            "Save annotated video",
            value=False,
            help="Save video with bounding boxes (slower)"
        )
    
    # Upload button
    st.divider()
    
    if uploaded_file is not None:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            if st.button("🚀 Start Processing", type="primary", width='stretch):
                # Upload and process
                with st.spinner("Uploading video to API..."):
                    result = upload_video_to_api(
                        api_base_url=api_base_url,
                        video_file=uploaded_file,
                        frame_width_meters=frame_width,
                        frame_height_meters=frame_height,
                        max_frames=max_frames,
                        save_output_video=save_output
                    )
                
                if result:
                    st.success("✅ Video uploaded successfully!")
                    st.balloons()
                    
                    # Store job_id in session state for monitoring
                    st.session_state['current_job_id'] = result['job_id']
                    st.session_state['current_job_filename'] = uploaded_file.name
                    
                    # Display job info
                    st.info(f"**Job ID:** `{result['job_id']}`")
                    st.info(f"**Status:** {result['status']}")
                    
                    # Prompt to switch to Monitor tab
                    st.success("👉 Switch to the **Monitor** tab to watch live progress!")
                else:
                    st.error("❌ Upload failed. Check API status.")
        
        with col_btn2:
            if st.button("🔄 Clear", width='stretch):
                st.rerun()
    
    else:
        st.info("👆 Please upload a video file to begin")


def upload_video_to_api(
    api_base_url: str,
    video_file,
    frame_width_meters: float,
    frame_height_meters: float,
    max_frames: int = None,
    save_output_video: bool = False
) -> dict:
    """
    Upload video file to FastAPI backend.
    
    Args:
        api_base_url: Base URL of API
        video_file: Uploaded file object from Streamlit
        frame_width_meters: Camera calibration width
        frame_height_meters: Camera calibration height
        max_frames: Optional max frames to process
        save_output_video: Whether to save annotated output
    
    Returns:
        dict: API response with job_id and status
    """
    try:
        # Prepare the file for upload
        files = {
            'file': (video_file.name, video_file.getvalue(), video_file.type)
        }
        
        # Prepare form data
        data = {
            'frame_width_meters': frame_width_meters,
            'frame_height_meters': frame_height_meters,
            'save_output_video': save_output_video
        }
        
        if max_frames is not None:
            data['max_frames'] = max_frames
        
        # Make API request
        response = requests.post(
            f"{api_base_url}/api/process",
            files=files,
            data=data,
            timeout=30  # 30 second timeout for upload
        )
        
        response.raise_for_status()  # Raise error for bad status codes
        
        return response.json()
    
    except requests.exceptions.Timeout:
        st.error("⏱️ Upload timeout. Video might be too large.")
        return None
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ API Error: {str(e)}")
        return None
    
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None