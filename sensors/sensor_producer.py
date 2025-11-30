import json
import random
import time
from datetime import datetime
from kafka import KafkaProducer

# Initialize Kafka producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# List of farm locations
farm_zones = ["North Field", "South Greenhouse", "East Orchard", "West Pasture"]

def generate_sensor_data():
    """Generate random sensor readings."""
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone": random.choice(farm_zones),
        "temperature": round(random.uniform(18.0, 35.0), 2),
        "humidity": round(random.uniform(30.0, 80.0), 2),
        "soil_moisture": round(random.uniform(200, 800), 2),
        "light_intensity": round(random.uniform(300, 1000), 2)
    }
    return data

if __name__ == "__main__":
    print("🚜 Starting IoT sensor data producer... (Press Ctrl+C to stop)")
    while True:
        try:
            data = generate_sensor_data()
            producer.send('sensor_data', value=data)
            print(f"✅ Sent: {data}")
            time.sleep(2)  # send data every 2 seconds
        except KeyboardInterrupt:
            print("\n🛑 Stopped producer manually.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")
            time.sleep(5)

