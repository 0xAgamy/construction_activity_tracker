import json
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable

KAFKA_BOOTSTRAP = "localhost:9092"
KAFKA_TOPIC     =  "equipment-events"

def create_kafka_producer(retries: int = 10, delay: int = 5) -> KafkaProducer:
    """A Function that create kafka producer

    Return:
        KafkaProducer
    """
    for attempt in range(retries):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BOOTSTRAP],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Reliability settings
                acks="all",
                retries=3,
                linger_ms=10,        # Small batching window
                batch_size=16384
            )
            print(f"Kafka producer connected to {KAFKA_BOOTSTRAP}")
            return producer
        except NoBrokersAvailable:
            print(f"Kafka not ready (attempt {attempt+1}/{retries}). "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Could not connect to Kafka after retries.")

def create_kafka_consumer()-> KafkaConsumer:
    """A Function that create kafka Consumer
    
    Return:
    KafkaConsumer
    """
    consumer = KafkaConsumer(
         KAFKA_TOPIC,
        bootstrap_servers=[KAFKA_BOOTSTRAP],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True
    )
    return consumer
    

