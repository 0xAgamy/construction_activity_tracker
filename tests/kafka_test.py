# test_kafka.py
from kafka import KafkaProducer, KafkaConsumer
import json
import time

TOPIC = "test-topic"

def test_connection():
    print("kafka runds...")
    
    #  Create Producer
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    #  Send a message
    msg = {"status": "pipeline_alive", "timestamp": time.time()}
    print(f"🔹 Sending: {msg}")
    
    try:
        future = producer.send(TOPIC, value=msg)
        record_metadata = future.get(timeout=10)  # Wait for confirmation
        print(f"Message sent to topic: {record_metadata.topic}")
    except Exception as e:
        print(f" Send failed: {e}")
        return
    
    producer.flush()
    
    
    time.sleep(2)
    
    #  Create Consumer 
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',  
        consumer_timeout_ms=10000,     
        group_id='test-group'          
    )
    
    # 5. Try to receive
    received = False
    print("Listening for messages...")
    for message in consumer:
        data = json.loads(message.value.decode('utf-8'))
        print(f"Received: {data}")
        received = True
        break
        
    consumer.close()
    
    if not received:
        print("Failed to receive message.")
   
    else:
        print("Kafka is running Successfully!")

if __name__ == "__main__":
    test_connection()