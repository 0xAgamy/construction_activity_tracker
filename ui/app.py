import gradio as gr
import pandas as pd
import psycopg2
import cv2
import numpy as np
import os
import time
import redis
import logging
import shutil
from datetime import datetime
import sys

# Add services to path for imports
sys.path.append('/app')

# ── Logging Setup ────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "construction_db")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_FRAME_KEY = "latest_frame"

INPUTS_DIR = "/app/inputs"
OUTPUTS_DIR = "/app/outputs"

# ── Import CV Engine (Direct Call) ───────────────────────────
try:
    from services.cv_engine.object_tracker import process_video
    logger.info("✅ CV Engine imported successfully")
except Exception as e:
    logger.warning(f"⚠️ CV Engine import failed: {e}")
    process_video = None

# ── Redis Client ─────────────────────────────────────────────
redis_client = None
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_timeout=2)
    redis_client.ping()
    logger.info(f"✅ Redis connected at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    logger.warning(f"⚠️ Redis connection failed: {e}")

# ── Database Connection ──────────────────────────────────────
def get_db_connection():
    try:
        return psycopg2.connect(DB_URI, connect_timeout=5)
    except Exception as e:
        logger.error(f"❌ DB connection failed: {e}")
        return None

# ── Video Processing Function ────────────────────────────────
def process_uploaded_video(video_path):
    """Process uploaded video using CV Engine (Direct Call)"""
    if video_path is None:
        return "❌ No video uploaded", None, "No data", []
    
    if process_video is None:
        return "❌ CV Engine not available", None, "No data", []
    
    try:
        # 1. Copy video to inputs directory
        video_filename = os.path.basename(video_path)
        dest_path = os.path.join(INPUTS_DIR, video_filename)
        shutil.copy(video_path, dest_path)
        logger.info(f"📁 Video copied to {dest_path}")
        
        # 2. Run CV Engine directly
        logger.info("🚀 Starting CV Engine processing...")
        process_video(dest_path)
        
        # 3. Get output video
        output_video = os.path.join(OUTPUTS_DIR, "output.mp4")
        
        if os.path.exists(output_video):
            return "✅ Processing complete!", output_video, get_analysis_summary(), get_latest_table()
        else:
            return "⚠️ Processing done but no output video found", None, get_analysis_summary(), get_latest_table()
            
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None, "Error getting summary", []

def get_analysis_summary():
    """Get summary statistics from database"""
    conn = get_db_connection()
    if not conn:
        return "❌ DB connection failed"
    
    try:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM equipment_events")
        total_records = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(DISTINCT track_id) FROM equipment_events")
        total_tracks = cur.fetchone()[0]
        
        cur.execute("""
            SELECT 
                SUM(active_sec) as total_active,
                SUM(idle_sec) as total_idle,
                AVG(util_pct) as avg_util
            FROM (
                SELECT DISTINCT ON (track_id) active_sec, idle_sec, util_pct
                FROM equipment_events
                ORDER BY track_id, time DESC
            ) AS latest
        """)
        row = cur.fetchone()
        
        cur.close()
        conn.close()
        
        return f"""
        ### 📊 Analysis Summary
        - **Total Records:** {total_records}
        - **Tracks Detected:** {total_tracks}
        - **Total Active Time:** {row[0]:.1f}s
        - **Total Idle Time:** {row[1]:.1f}s
        - **Average Utilization:** {row[2]:.1f}%
        """
    except Exception as e:
        return f"❌ Error getting summary: {e}"

def get_latest_table():
    """Get latest machine status from database"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        cur = conn.cursor()
        
        query = """
            SELECT *
            FROM (
                SELECT DISTINCT ON (track_id)
                    track_id, class_name, state, activity, util_pct, 
                    active_sec, idle_sec, time
                FROM equipment_events
                ORDER BY track_id, time DESC
            ) AS latest_events
            ORDER BY time DESC;
        """
        
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if rows:
            df = pd.DataFrame(rows, columns=['track_id', 'class_name', 'state', 'activity', 
                                            'util_pct', 'active_sec', 'idle_sec', 'time'])
            df['util_pct'] = df['util_pct'].round(2).astype(str) + '%'
            df['active_sec'] = df['active_sec'].round(1).astype(str) + 's'
            df['idle_sec'] = df['idle_sec'].round(1).astype(str) + 's'
            df['time'] = df['time'].apply(lambda x: x.strftime('%H:%M:%S') if hasattr(x, 'strftime') else str(x))
            return df[['track_id', 'class_name', 'state', 'activity', 'active_sec', 'idle_sec', 'util_pct', 'time']].values.tolist()
        else:
            return []
    except Exception as e:
        logger.error(f"❌ Table query error: {e}")
        return []

# ── Gradio Interface (Single Page) ───────────────────────────
with gr.Blocks(title="Construction Equipment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚜 Construction Equipment Utilization Analyzer")
    gr.Markdown("### Upload a video to analyze equipment activity and utilization")
    
    # ── Upload & Process Section ─────────────────────────────
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="📹 Upload Video", sources=["upload", "webcam"])
            process_btn = gr.Button("🚀 Process Video", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Processing Status", lines=2)
            video_output = gr.Video(label="🎥 Processed Video with Annotations")
    
    # ── Results Section ──────────────────────────────────────
    gr.Markdown("## 📊 Analysis Results")
    
    with gr.Row():
        summary_output = gr.Markdown(label="Summary")
    
    with gr.Row():
        table_output = gr.Dataframe(
            headers=["Track ID", "Class", "State", "Activity", "Active", "Idle", "Util %", "Time"],
            label="📋 Machine Status",
            wrap=True,
            interactive=False
        )
    
    # ── Process Button Action ────────────────────────────────
    process_btn.click(
        fn=process_uploaded_video,
        inputs=[video_input],
        outputs=[status_output, video_output, summary_output, table_output]
    )
    
    # ── Footer ───────────────────────────────────────────────
    gr.Markdown("---")

if __name__ == "__main__":
    logger.info("🚀 Starting Gradio UI...")
    demo.launch(server_name="0.0.0.0", server_port=7860)