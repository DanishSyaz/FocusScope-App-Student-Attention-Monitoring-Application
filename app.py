import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import math
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime
import time
import io
from collections import deque

# Try to import pynput for real input tracking
try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

# --- CONFIGURATION & OPERATIONAL DEFINITIONS ---
# FR1: User Presence Detection
FACE_DETECTION_CONFIDENCE = 0.5
PRESENCE_CONFIRMATION_FRAMES = 3

# FR2 & FR3: Head Pose & Gaze Tracking Thresholds
HEAD_YAW_THRESHOLD = 80
HEAD_PITCH_THRESHOLD = 60
GAZE_AWAY_DURATION_THRESHOLD = 2.0

# FR4: Input Activity Monitoring
INPUT_IDLE_THRESHOLD = 3.0

# FR5: Distraction Classification
DISTRACTION_HEAD_AWAY_DURATION = 3.0
FOCUS_CONFIRMATION_DURATION = 2.0
IDLE_NO_PRESENCE_DURATION = 5.0

# FR6: Real-time Status Indicator
STATUS_INDICATOR_HEIGHT = 50
STATUS_INDICATOR_OPACITY = 0.8

# Camera settings
CAMERA_FPS = 30
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(255, 255, 255))

# Landmark indices
NOSE_TIP = 1
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
CHIN = 152
FOREHEAD = 10

# Input tracking global state
class InputTracker:
    """Real-time keyboard and mouse input tracker using pynput"""
    def __init__(self):
        self.keyboard_events = deque(maxlen=100)
        self.mouse_events = deque(maxlen=100)
        self.keyboard_listener = None
        self.mouse_listener = None
        self.running = False
        
    def on_keyboard_event(self, key):
        """Callback for keyboard events"""
        self.keyboard_events.append(time.time())
        
    def on_mouse_move(self, x, y):
        """Callback for mouse movement"""
        self.mouse_events.append(time.time())
        
    def on_mouse_click(self, x, y, button, pressed):
        """Callback for mouse clicks"""
        if pressed:
            self.mouse_events.append(time.time())
    
    def start(self):
        """Start listening to input events"""
        if not PYNPUT_AVAILABLE or self.running:
            return
            
        try:
            # Start keyboard listener
            self.keyboard_listener = keyboard.Listener(
                on_press=self.on_keyboard_event,
                on_release=self.on_keyboard_event
            )
            self.keyboard_listener.start()
            
            # Start mouse listener
            self.mouse_listener = mouse.Listener(
                on_move=self.on_mouse_move,
                on_click=self.on_mouse_click
            )
            self.mouse_listener.start()
            
            self.running = True
        except Exception as e:
            print(f"Error starting input tracking: {e}")
    
    def stop(self):
        """Stop listening to input events"""
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        if self.mouse_listener:
            self.mouse_listener.stop()
        self.running = False
        
    def get_activity(self, time_window: float = 2.0) -> Dict[str, bool]:
        """
        Check if there was input activity in the last time_window seconds
        """
        current_time = time.time()
        
        # Check keyboard activity
        keyboard_active = any(
            current_time - event_time < time_window 
            for event_time in self.keyboard_events
        )
        
        # Check mouse activity
        mouse_active = any(
            current_time - event_time < time_window 
            for event_time in self.mouse_events
        )
        
        return {
            'has_keyboard': keyboard_active,
            'has_mouse': mouse_active,
            'has_any_input': keyboard_active or mouse_active
        }
    
    def clear(self):
        """Clear all stored events"""
        self.keyboard_events.clear()
        self.mouse_events.clear()

# Global input tracker instance
input_tracker = InputTracker()

def get_input_activity(is_present: bool) -> Dict[str, any]:
    """FR4: Input Activity Monitoring using pynput"""
    if PYNPUT_AVAILABLE:
        # REAL MODE: Use actual input tracking
        return input_tracker.get_activity(time_window=2.0)
    else:
        # Fallback if pynput not available
        return {
            'has_keyboard': False,
            'has_mouse': False,
            'has_any_input': False
        }

def get_landmark_coords(landmarks, index: int, width: int, height: int) -> Tuple[int, int]:
    """Retrieves normalized landmark coordinates and scales them to frame size."""
    if not landmarks or index >= len(landmarks.landmark):
        return (0, 0)
    
    point = landmarks.landmark[index]
    return (int(point.x * width), int(point.y * height))

def estimate_head_pose(face_landmarks, width: int, height: int) -> Dict[str, any]:
    """FR2: Head Pose Tracking"""
    nose_tip = get_landmark_coords(face_landmarks, NOSE_TIP, width, height)
    
    center_x = width // 2
    center_y = height // 2
    
    yaw_offset = nose_tip[0] - center_x
    pitch_offset = nose_tip[1] - center_y
    
    looking_at_screen = (
        abs(yaw_offset) < HEAD_YAW_THRESHOLD and 
        abs(pitch_offset) < HEAD_PITCH_THRESHOLD
    )
    
    direction = "Center"
    if abs(yaw_offset) > HEAD_YAW_THRESHOLD:
        direction = "Right" if yaw_offset > 0 else "Left"
    elif abs(pitch_offset) > HEAD_PITCH_THRESHOLD:
        direction = "Down" if pitch_offset > 0 else "Up"
    
    return {
        'yaw_offset': yaw_offset,
        'pitch_offset': pitch_offset,
        'looking_at_screen': looking_at_screen,
        'direction': direction,
        'nose_tip': nose_tip
    }

def infer_gaze_direction(face_landmarks, width: int, height: int) -> Dict[str, any]:
    """FR3: Basic Gaze Inference"""
    left_eye = get_landmark_coords(face_landmarks, LEFT_EYE_OUTER, width, height)
    right_eye = get_landmark_coords(face_landmarks, RIGHT_EYE_OUTER, width, height)
    
    eye_center_x = (left_eye[0] + right_eye[0]) // 2
    eye_center_y = (left_eye[1] + right_eye[1]) // 2
    
    center_x = width // 2
    center_y = height // 2
    
    gaze_x_offset = eye_center_x - center_x
    gaze_y_offset = eye_center_y - center_y
    
    gaze_towards_screen = (
        abs(gaze_x_offset) < HEAD_YAW_THRESHOLD and 
        abs(gaze_y_offset) < HEAD_PITCH_THRESHOLD
    )
    
    return {
        'gaze_x_offset': gaze_x_offset,
        'gaze_y_offset': gaze_y_offset,
        'gaze_towards_screen': gaze_towards_screen,
        'eye_center': (eye_center_x, eye_center_y)
    }

class AttentionStateTracker:
    """FR5: Distraction Classification"""
    def __init__(self, fps: float):
        self.fps = fps
        self.consecutive_looking_away = 0
        self.consecutive_no_presence = 0
        self.consecutive_focused = 0
        self.last_input_time = 0
        self.current_frame = 0
        
    def update(self, is_present: bool, looking_at_screen: bool, 
               has_input: bool, gaze_towards_screen: bool) -> str:
        """
        Classifies current attention state.
        Returns: status string
        """
        self.current_frame += 1
        current_time = self.current_frame / self.fps
        
        if has_input:
            self.last_input_time = current_time
        
        time_since_input = current_time - self.last_input_time
        
        # IDLE: User not present
        if not is_present:
            self.consecutive_no_presence += 1
            self.consecutive_looking_away = 0
            self.consecutive_focused = 0
            
            if (self.consecutive_no_presence / self.fps) >= IDLE_NO_PRESENCE_DURATION:
                return "IDLE"
            else:
                return "IDLE (Temporary)"
        
        self.consecutive_no_presence = 0
        
        # DISTRACTED: Head turned away
        if not looking_at_screen or not gaze_towards_screen:
            self.consecutive_looking_away += 1
            self.consecutive_focused = 0
            
            duration_looking_away = self.consecutive_looking_away / self.fps
            
            if duration_looking_away >= DISTRACTION_HEAD_AWAY_DURATION:
                if time_since_input > INPUT_IDLE_THRESHOLD:
                    return "DISTRACTED (Not Engaged)"
                else:
                    return "DISTRACTED (Multi-tasking)"
            else:
                return "DISTRACTED (Brief)"
        
        # FOCUSED
        self.consecutive_looking_away = 0
        self.consecutive_focused += 1
        
        duration_focused = self.consecutive_focused / self.fps
        
        if duration_focused >= FOCUS_CONFIRMATION_DURATION:
            if has_input or time_since_input <= INPUT_IDLE_THRESHOLD:
                return "FOCUSED (Active)"
            else:
                return "FOCUSED (Passive/Reading)"
        else:
            return "FOCUSED (Engaging)"

def draw_status_indicator(frame: np.ndarray, status: str, 
                         pose_info: Dict, width: int, height: int) -> np.ndarray:
    """FR6: Real-time Status Indicator"""
    status_colors = {
        "FOCUSED": (0, 255, 0),
        "DISTRACTED": (0, 165, 255),
        "IDLE": (128, 128, 128)
    }
    
    primary_status = status.split()[0]
    color = status_colors.get(primary_status, (0, 0, 0))
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height - STATUS_INDICATOR_HEIGHT), 
                  (width, height), color, -1)
    cv2.addWeighted(overlay, STATUS_INDICATOR_OPACITY, frame, 
                   1 - STATUS_INDICATOR_OPACITY, 0, frame)
    
    cv2.putText(frame, status, (20, height - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    direction_text = f"Head: {pose_info.get('direction', 'Unknown')}"
    cv2.putText(frame, direction_text, (20, height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return frame

def process_frame(frame: np.ndarray, face_mesh, state_tracker: AttentionStateTracker) -> Dict:
    """Main frame processing with all FRs"""
    height, width, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    is_present = bool(results.multi_face_landmarks)
    
    pose_info = {'looking_at_screen': False, 'direction': 'Unknown', 'nose_tip': (0, 0)}
    gaze_info = {'gaze_towards_screen': False}
    input_info = {'has_any_input': False, 'has_keyboard': False, 'has_mouse': False}
    
    if is_present and results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        pose_info = estimate_head_pose(face_landmarks, width, height)
        gaze_info = infer_gaze_direction(face_landmarks, width, height)
        
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )
        
        center_x = width // 2
        center_y = height // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 255), 1)
        cv2.line(frame, (0, center_y), (width, center_y), (255, 255, 255), 1)
        cv2.circle(frame, pose_info['nose_tip'], 5, (255, 0, 255), -1)
    
    input_info = get_input_activity(is_present)
    
    status = state_tracker.update(
        is_present=is_present,
        looking_at_screen=pose_info['looking_at_screen'],
        has_input=input_info['has_any_input'],
        gaze_towards_screen=gaze_info['gaze_towards_screen']
    )
    
    frame = draw_status_indicator(frame, status, pose_info, width, height)
    
    return {
        'frame': frame,
        'is_present': is_present,
        'status': status,
        'looking_at_screen': pose_info['looking_at_screen'],
        'head_direction': pose_info['direction'],
        'gaze_towards_screen': gaze_info['gaze_towards_screen'],
        'has_input': input_info['has_any_input'],
        'has_keyboard': input_info['has_keyboard'],
        'has_mouse': input_info['has_mouse']
    }

def save_session_to_csv(metrics: List[Dict], filename: str = "attention_history.csv"):
    """Save current session metrics to CSV file (append mode)"""
    df = pd.DataFrame(metrics)
    
    # Add session metadata
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    df['session_id'] = session_id
    df['timestamp'] = pd.to_datetime(datetime.now()) + pd.to_timedelta(df['time_sec'], unit='s')
    
    # Reorder columns
    columns_order = ['session_id', 'timestamp', 'time_sec', 'status', 'is_present', 
                     'looking_at_screen', 'head_direction', 'gaze_towards_screen',
                     'has_input', 'has_keyboard', 'has_mouse']
    df = df[columns_order]
    
    # Append to existing file or create new
    try:
        existing_df = pd.read_csv(filename)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(filename, index=False)
    except FileNotFoundError:
        df.to_csv(filename, index=False)
    
    return filename

def analyze_historical_data(df: pd.DataFrame):
    """Analyze and visualize historical attention data"""
    st.header("📈 Historical Attention Trend Analysis")
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['session_id'] = df['session_id'].astype(str)
    
    # Get unique sessions
    sessions = df['session_id'].unique()
    
    st.subheader(f"📊 Total Sessions: {len(sessions)}")
    
    # Session-by-session analysis
    session_summary = []
    for session_id in sessions:
        session_df = df[df['session_id'] == session_id]
        
        # Calculate session metrics
        total_time = session_df['time_sec'].max()
        fps = 30  # Assume 30 FPS
        
        df_primary = session_df.copy()
        df_primary['primary_status'] = df_primary['status'].apply(lambda x: x.split()[0])
        
        focused_count = len(df_primary[df_primary['primary_status'] == 'FOCUSED'])
        distracted_count = len(df_primary[df_primary['primary_status'] == 'DISTRACTED'])
        idle_count = len(df_primary[df_primary['primary_status'] == 'IDLE'])
        
        focused_time = focused_count / fps
        distracted_time = distracted_count / fps
        idle_time = idle_count / fps
        
        attention_score = (focused_time / total_time * 100) if total_time > 0 else 0
        
        session_summary.append({
            'Session ID': session_id,
            'Date': session_df['timestamp'].min().strftime('%Y-%m-%d %H:%M'),
            'Duration (s)': round(total_time, 1),
            'Focused (s)': round(focused_time, 1),
            'Distracted (s)': round(distracted_time, 1),
            'Idle (s)': round(idle_time, 1),
            'Attention Score': f"{attention_score:.1f}%"
        })
    
    summary_df = pd.DataFrame(session_summary)
    st.dataframe(summary_df, hide_index=True, use_container_width=True)
    
    # Trend over time
    st.subheader("📉 Attention Score Trend")
    
    # Calculate attention score per session for line chart
    trend_data = []
    for i, session_id in enumerate(sessions):
        session_df = df[df['session_id'] == session_id]
        
        df_primary = session_df.copy()
        df_primary['primary_status'] = df_primary['status'].apply(lambda x: x.split()[0])
        focused_count = len(df_primary[df_primary['primary_status'] == 'FOCUSED'])
        
        attention_score = (focused_count / len(session_df) * 100) if len(session_df) > 0 else 0
        
        trend_data.append({
            'Session': i + 1,
            'Attention Score': attention_score,
            'Date': session_df['timestamp'].min().strftime('%m/%d %H:%M')
        })
    
    trend_df = pd.DataFrame(trend_data)
    st.line_chart(trend_df.set_index('Session')['Attention Score'])
    
    # Show session details
    st.subheader("🔍 Detailed Session Information")
    
    for session_id in sessions[-5:]:  # Show last 5 sessions
        session_df = df[df['session_id'] == session_id]
        
        with st.expander(f"📅 Session: {session_df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')}"):
            # Timeline for this session
            session_df['time_bin'] = (session_df['time_sec'] // 1).astype(int)
            session_df['primary_status'] = session_df['status'].apply(lambda x: x.split()[0])
            session_df['attention_score'] = session_df['primary_status'].map({
                'FOCUSED': 1.0,
                'DISTRACTED': 0.3,
                'IDLE': 0.0
            })
            
            timeline_df = session_df.groupby('time_bin')['attention_score'].mean().reset_index()
            timeline_df.columns = ['Time (s)', 'Attention Score']
            
            st.line_chart(timeline_df.set_index('Time (s)'))
            
            # Key events in this session
            st.markdown("**Key Events:**")
            
            # Find distraction periods
            distractions = session_df[session_df['primary_status'] == 'DISTRACTED']
            if not distractions.empty:
                distraction_times = distractions.groupby((distractions['time_sec'] // 5).astype(int))['time_sec'].agg(['min', 'max'])
                
                for _, row in distraction_times.head(5).iterrows():
                    st.write(f"• Distraction detected: {row['min']:.1f}s - {row['max']:.1f}s")
            
            # Head direction summary
            direction_summary = session_df['head_direction'].value_counts()
            st.write(f"**Most common head direction:** {direction_summary.index[0]} ({direction_summary.values[0]} frames)")
    
    # Overall insights
    st.subheader("💡 Insights & Patterns")
    
    # Calculate overall statistics
    df['primary_status'] = df['status'].apply(lambda x: x.split()[0])
    overall_focused = len(df[df['primary_status'] == 'FOCUSED']) / len(df) * 100
    overall_distracted = len(df[df['primary_status'] == 'DISTRACTED']) / len(df) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Overall Focus Rate", f"{overall_focused:.1f}%")
        st.metric("Overall Distraction Rate", f"{overall_distracted:.1f}%")
    
    with col2:
        # Time of day analysis
        df['hour'] = df['timestamp'].dt.hour
        peak_focus_hour = df[df['primary_status'] == 'FOCUSED'].groupby('hour').size().idxmax() if not df[df['primary_status'] == 'FOCUSED'].empty else "N/A"
        st.metric("Peak Focus Hour", f"{peak_focus_hour}:00" if peak_focus_hour != "N/A" else "N/A")
        
        # Average session duration
        avg_duration = summary_df['Duration (s)'].apply(lambda x: float(x)).mean()
        st.metric("Avg Session Duration", f"{avg_duration:.1f}s")
    
    # Recommendations based on historical data
    st.subheader("📋 Personalized Recommendations")
    
    if overall_focused < 50:
        st.warning("⚠️ Your overall focus rate is below 50%. Consider:")
        st.write("• Breaking work into smaller sessions (Pomodoro technique)")
        st.write("• Identifying and eliminating common distractions")
        st.write("• Optimizing your workspace environment")
    elif overall_focused < 70:
        st.info("ℹ️ Good focus rate! To improve further:")
        st.write("• Maintain consistent work schedule")
        st.write("• Take regular breaks to prevent fatigue")
    else:
        st.success("🌟 Excellent focus! Keep up the good work!")
        st.write("• Continue your current practices")
        st.write("• Share your successful strategies with others")

def identify_distraction_triggers(df: pd.DataFrame, fps: float) -> List[Dict]:
    """Identifies distraction triggers"""
    triggers = []
    
    for i in range(1, len(df)):
        prev_status = df.iloc[i-1]['status']
        curr_status = df.iloc[i]['status']
        
        if 'FOCUSED' in prev_status and 'DISTRACTED' in curr_status:
            triggers.append({
                'time': df.iloc[i]['time_sec'],
                'type': 'Focus Lost',
                'from_state': prev_status,
                'to_state': curr_status,
                'had_input': df.iloc[i]['has_input']
            })
        
        if 'DISTRACTED (Not Engaged)' in curr_status:
            if i == 0 or 'DISTRACTED (Not Engaged)' not in prev_status:
                triggers.append({
                    'time': df.iloc[i]['time_sec'],
                    'type': 'Sustained Distraction',
                    'from_state': prev_status,
                    'to_state': curr_status,
                    'had_input': df.iloc[i]['has_input']
                })
    
    return triggers

def generate_personalized_report(df: pd.DataFrame, fps: float, 
                                 total_duration_sec: float, triggers: List[Dict]):
    """FR7: Personalized Report Generation"""
    st.header("📊 Session Report")
    
    df['primary_status'] = df['status'].apply(lambda x: x.split()[0])
    
    status_times = {}
    for status in ['FOCUSED', 'DISTRACTED', 'IDLE']:
        count = len(df[df['primary_status'] == status])
        status_times[status] = count / fps
    
    # Summary Metrics
    st.subheader("📈 Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Duration", f"{total_duration_sec:.1f}s")
    
    with col2:
        focused_pct = (status_times['FOCUSED'] / total_duration_sec) * 100
        st.metric("Total Focused Time", 
                 f"{status_times['FOCUSED']:.1f}s",
                 f"{focused_pct:.1f}%")
    
    with col3:
        distracted_pct = (status_times['DISTRACTED'] / total_duration_sec) * 100
        st.metric("Total Distracted Time", 
                 f"{status_times['DISTRACTED']:.1f}s",
                 f"{distracted_pct:.1f}%")
    
    with col4:
        idle_pct = (status_times['IDLE'] / total_duration_sec) * 100
        st.metric("Total Idle Time", 
                 f"{status_times['IDLE']:.1f}s",
                 f"{idle_pct:.1f}%")
    
    # Attention Score
    st.subheader("🎯 Attention Score")
    attention_score = (status_times['FOCUSED'] / total_duration_sec) * 100
    
    score_col1, score_col2 = st.columns([1, 2])
    with score_col1:
        st.metric("Score", f"{attention_score:.0f}/100")
    
    with score_col2:
        if attention_score >= 75:
            st.success("🌟 Excellent! Strong focus throughout the session.")
        elif attention_score >= 60:
            st.info("👍 Good focus. Minor distractions detected.")
        elif attention_score >= 40:
            st.warning("⚠️ Moderate distractions. Consider environment optimization.")
        else:
            st.error("❗ High distraction level. Review triggers below.")
    
    # Timeline
    st.subheader("📈 Attention Timeline")
    
    df['time_bin'] = (df['time_sec'] // 1).astype(int)
    df['attention_score'] = df['primary_status'].map({
        'FOCUSED': 1.0,
        'DISTRACTED': 0.3,
        'IDLE': 0.0
    })
    
    timeline_df = df.groupby('time_bin')['attention_score'].mean().reset_index()
    timeline_df.columns = ['Time (s)', 'Attention Score']
    
    st.line_chart(timeline_df.set_index('Time (s)'))
    
    # Distraction Triggers
    st.subheader("⚠️ Distraction Events")
    
    if triggers:
        st.warning(f"**{len(triggers)} distraction events detected**")
        
        trigger_df = pd.DataFrame(triggers)
        trigger_df['Time (s)'] = trigger_df['time'].round(1)
        trigger_df = trigger_df[['Time (s)', 'type', 'from_state', 'to_state']]
        trigger_df.columns = ['Time (s)', 'Event Type', 'From', 'To']
        
        st.dataframe(trigger_df.head(10), hide_index=True, use_container_width=True)
    else:
        st.success("✅ No significant distraction triggers detected!")

# --- MAIN APPLICATION ---

st.set_page_config(
    page_title="Real-time Attention Monitor", 
    page_icon="📊", 
    layout="wide"
)

st.title("📊 Real-time Attention Monitoring System")

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'show_report' not in st.session_state:
    st.session_state.show_report = False
if 'csv_filename' not in st.session_state:
    st.session_state.csv_filename = "attention_history.csv"

# Check pynput availability
if not PYNPUT_AVAILABLE:
    st.warning("⚠️ **pynput library not installed.** Input tracking will not work. Install it with: `pip install pynput`")

# Tabs for different features
tab1, tab2 = st.tabs(["🎥 Live Monitoring", "📊 Historical Analysis"])

with tab1:
    st.markdown("### Monitor your attention in real-time")
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        if st.button("🎥 Start Monitoring", disabled=st.session_state.monitoring, 
                     use_container_width=True, type="primary"):
            st.session_state.monitoring = True
            st.session_state.metrics = []
            st.session_state.start_time = time.time()
            st.session_state.show_report = False
            st.rerun()
    
    with col_btn2:
        if st.button("⏹️ Stop & Save", disabled=not st.session_state.monitoring,
                     use_container_width=True):
            st.session_state.monitoring = False
            st.session_state.show_report = True
            st.rerun()
    
    # Monitoring status
    if st.session_state.monitoring:
        st.success("🔴 **MONITORING IN PROGRESS** - Click 'Stop' to save and generate report")
    elif st.session_state.show_report:
        st.info("📊 **Monitoring Complete** - Report generated below")
    else:
        st.info("👆 Click 'Start Monitoring' to begin tracking your attention")
    
    # Real-time monitoring
    if st.session_state.monitoring:
        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🎥 Live Feed")
            frame_placeholder = st.empty()
        
        with col2:
            st.markdown("### 📊 Current Status")
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
        
        # Start input tracking if available
        if PYNPUT_AVAILABLE and not input_tracker.running:
            input_tracker.start()
            st.success("✅ Real input tracking started")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            st.error("❌ Unable to access webcam. Please check permissions.")
            st.session_state.monitoring = False
            input_tracker.stop()
            st.stop()
        
        # Initialize tracking
        state_tracker = AttentionStateTracker(CAMERA_FPS)
        frame_count = 0
        
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECTION_CONFIDENCE,
            min_tracking_confidence=0.5
        ) as face_mesh:
            
            while st.session_state.monitoring:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read from webcam")
                    break
                
                frame_count += 1
                elapsed_time = time.time() - st.session_state.start_time
                
                # Process frame
                result = process_frame(frame, face_mesh, state_tracker)
                
                # Store metrics
                st.session_state.metrics.append({
                    'time_sec': elapsed_time,
                    'status': result['status'],
                    'is_present': result['is_present'],
                    'looking_at_screen': result['looking_at_screen'],
                    'head_direction': result['head_direction'],
                    'gaze_towards_screen': result['gaze_towards_screen'],
                    'has_input': result['has_input'],
                    'has_keyboard': result['has_keyboard'],
                    'has_mouse': result['has_mouse']
                })
                
                # Display frame
                frame_rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update status
                with status_placeholder.container():
                    st.metric("Attention State", result['status'])
                    st.metric("Head Direction", result['head_direction'])
                    present_status = "✅ Present" if result['is_present'] else "❌ Absent"
                    st.metric("User Presence", present_status)
                
                with metrics_placeholder.container():
                    st.metric("Duration", f"{elapsed_time:.1f}s")
                    st.metric("Frames", frame_count)
                    input_status = "🟢 Active" if result['has_input'] else "⚪ Idle"
                    st.metric("Input", input_status)
                
                # Small delay to control frame rate
                time.sleep(1/CAMERA_FPS)
        
        cap.release()
        input_tracker.stop()
    
    # Generate report and save to CSV if stopped
    if st.session_state.show_report and len(st.session_state.metrics) > 0:
        df = pd.DataFrame(st.session_state.metrics)
        total_duration = st.session_state.metrics[-1]['time_sec']
        triggers = identify_distraction_triggers(df, CAMERA_FPS)
        
        # Save to CSV
        csv_filename = save_session_to_csv(st.session_state.metrics, st.session_state.csv_filename)
        st.success(f"✅ Session data saved to `{csv_filename}`")
        
        # Download button for CSV
        csv_data = pd.read_csv(csv_filename)
        csv_buffer = io.StringIO()
        csv_data.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📥 Download Complete History (CSV)",
            data=csv_buffer.getvalue(),
            file_name=csv_filename,
            mime="text/csv",
            use_container_width=True
        )
        
        # Generate report
        generate_personalized_report(df, CAMERA_FPS, total_duration, triggers)
        
        # Option to start new session
        st.markdown("---")
        if st.button("🔄 Start New Monitoring Session", type="primary"):
            st.session_state.monitoring = False
            st.session_state.metrics = []
            st.session_state.show_report = False
            st.rerun()

with tab2:
    st.markdown("### Upload your CSV file to analyze attention trends over time")
    
    # File uploader for CSV
    uploaded_csv = st.file_uploader(
        "📁 Upload Attention History CSV", 
        type=['csv'],
        help="Upload your saved attention_history.csv file to view historical trends"
    )
    
    if uploaded_csv is not None:
        try:
            df_history = pd.read_csv(uploaded_csv)
            
            # Validate CSV structure
            required_columns = ['session_id', 'timestamp', 'time_sec', 'status', 'is_present']
            if not all(col in df_history.columns for col in required_columns):
                st.error("❌ Invalid CSV format. Please upload a valid attention_history.csv file.")
            else:
                st.success(f"✅ Loaded {len(df_history)} records from {len(df_history['session_id'].unique())} sessions")
                
                # Analyze historical data
                analyze_historical_data(df_history)
                
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
    else:
        st.info("👆 Upload your CSV file to view historical attention trends and patterns")
        
        # Show sample of what the analysis looks like
        with st.expander("ℹ️ What you'll see in Historical Analysis"):
            st.markdown("""
            ### Historical Analysis Features:
            
            1. **Session Summary Table**
               - View all your monitoring sessions
               - See duration, focus time, and attention scores for each session
              
            2. **Attention Score Trend**
               - Line chart showing how your attention has improved over time
               - Identify patterns and improvements
             
            3. **Detailed Session Information**
               - Expandable view for each session
               - Timeline of attention states
               - Key distraction events and when they occurred
                        
            4. **Overall Insights**
               - Peak focus hours
               - Average session duration
               - Focus and distraction rates
                        
            5. **Personalized Recommendations**
               - Based on your historical data
               - Actionable tips to improve focus
            """)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### CSV File Settings")
    csv_filename = st.text_input(
        "CSV Filename",
        value=st.session_state.csv_filename,
        help="Name of the CSV file to save/append data"
    )
    st.session_state.csv_filename = csv_filename
    
    st.markdown("---")
    
    st.header("ℹ️ System Information")
    
    st.markdown("""
    ### Features
    - ✅ Real-time webcam monitoring
    - 🎭 Head pose tracking
    - 👁️ Gaze direction inference
    - 🎯 Attention classification
    - ⌨️ Real input tracking
    - 💾 CSV data persistence
    - 📈 Historical trend analysis
    - 📊 Comprehensive reports
    
    ### How to Use
    **Live Monitoring:**
    1. Click **Start Monitoring**
    2. Work normally while being monitored
    3. Click **Stop & Save** when finished
    4. Review report and download CSV
    
    **Historical Analysis:**
    1. Upload your saved CSV file
    2. View trends across sessions
    3. Identify patterns and improvements
    4. Get personalized recommendations
    
    ### Privacy Note
    All processing happens locally. No video is recorded or transmitted. Data is only saved to CSV on your device.
    """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Input Tracking
    - 🎹 Monitors real keyboard presses
    - 🖱️ Tracks actual mouse movements & clicks
    - 📊 Shows "Active" only when you interact
    - ⏱️ 2-second activity window
    """)
