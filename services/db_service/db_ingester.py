import json
import time
import psycopg2
from psycopg2.extras import execute_values
from kafka import KafkaConsumer
import os
import logging

# ── Logging Setup ────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "kafka:29092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "equipment-events")

DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "construction_db")
DB_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:5432/{DB_NAME}"

BATCH_SIZE = 50
COMMIT_INTERVAL = 10  # Commit every 10 seconds even if batch not full

# ── Insert Query ─────────────────────────────────────────────
INSERT_QUERY = """
    INSERT INTO equipment_events 
    (time, track_id, session_id, class_name, state, activity, confidence,
     bbox_x1, bbox_y1, bbox_x2, bbox_y2, active_sec, idle_sec, total_sec, util_pct)
    VALUES %s
"""

def run():
    logger.info("📦 Starting DB Ingester...")
    logger.info(f"Kafka: {KAFKA_BOOTSTRAP} → Topic: {KAFKA_TOPIC}")
    logger.info(f"DB: {DB_URI}")
    
    # Wait for Kafka to be ready
    time.sleep(5)
    
    # ── Kafka Consumer (Fixed: use poll() for better control) ─
    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        consumer_timeout_ms=10000,  # 10 seconds (we handle timeout manually)
        session_timeout_ms=30000,
        heartbeat_interval_ms=10000
    )
    
    logger.info(f"Kafka consumer subscribed to {KAFKA_TOPIC}")

    # ── Database Connection ───────────────────────────────────
    conn = None
    cur = None
    batch = []
    last_commit_time = time.time()
    message_count = 0
    
    try:
        conn = psycopg2.connect(DB_URI)
        cur = conn.cursor()
        logger.info(" Connected to TimescaleDB")

        # ── Main Consumer Loop (runs forever) ─────────────────
        while True:
            try:
                # Poll for messages with timeout
                msg_pack = consumer.poll(timeout_ms=1000)
                
                # Process messages
                for topic_partition, messages in msg_pack.items():
                    for msg in messages:
                        try:
                            payload = msg.value
                            
                            row = (
                                time.strftime("%Y-%m-%d %H:%M:%S+00", time.gmtime(payload["timestamp"])),
                                payload["track_id"],
                                payload["session_id"],
                                payload["class_name"],
                                payload["state"],
                                payload["activity"],
                                payload["confidence"],
                                payload["bbox"]["x1"], payload["bbox"]["y1"],
                                payload["bbox"]["x2"], payload["bbox"]["y2"],
                                payload["utilization"]["total_active_sec"],
                                payload["utilization"]["total_inactive_sec"],
                                payload["utilization"]["total_tracked_sec"],
                                payload["utilization"]["utilization_pct"]
                            )
                            batch.append(row)
                            message_count += 1

                            # Commit when batch is full
                            if len(batch) >= BATCH_SIZE:
                                execute_values(cur, INSERT_QUERY, batch)
                                conn.commit()
                                consumer.commit()
                                logger.info(f" Batch saved: {len(batch)} records (Total: {message_count})")
                                batch.clear()
                                last_commit_time = time.time()
                            
                        except Exception as e:
                            logger.error(f" Error processing message: {e}")
                            continue
                
                # Commit periodically even if batch not full
                if batch and time.time() - last_commit_time > COMMIT_INTERVAL:
                    execute_values(cur, INSERT_QUERY, batch)
                    conn.commit()
                    consumer.commit()
                    logger.info(f" Periodic commit: {len(batch)} records (Total: {message_count})")
                    batch.clear()
                    last_commit_time = time.time()
                
                # Heartbeat log to show ingester is alive
                if time.time() - last_commit_time > 10:
                    logger.info(f"Ingester alive - Waiting for messages... (Total: {message_count})")

            except KeyboardInterrupt:
                logger.info("\nIngester stopped by user.")
                break
            except Exception as e:
                logger.error(f"Consumer error: {e}")
                time.sleep(5)
                continue

    except KeyboardInterrupt:
        logger.info("\n Ingester stopped by user.")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ── Cleanup ───────────────────────────────────────────
        if batch and conn:
            try:
                execute_values(cur, INSERT_QUERY, batch)
                conn.commit()
                logger.info(f" Final commit: {len(batch)} records")
            except Exception as e:
                logger.error(f"Final commit failed: {e}")
        
        if cur:
            cur.close()
        if conn:
            conn.close()
        
        consumer.close()
        logger.info("DB Ingester shutdown complete.")

if __name__ == "__main__":
    run()