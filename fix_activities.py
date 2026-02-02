# Save this as fix_activities.py
import json
from pathlib import Path

data = [
    {
        "id": 1,
        "activity": "Jeep Safari",
        "prices": {"domestic": 500, "SAARC": 1500, "tourist": 3500},
        "schedule": "Morning/Evening",
        "timing": "(6-10) AM / (2-5) PM"
    },
    {
        "id": 2,
        "activity": "Elephant Safari",
        "prices": {"domestic": 1650, "SAARC": 4000, "tourist": 5000},
        "schedule": "Morning/Evening",
        "timing": "(6-10) AM / (2-5) PM"
    },
    {
        "id": 3,
        "activity": "Bird Watching",
        "prices": {"domestic": 3000, "SAARC": 5500, "tourist": 6500},
        "schedule": "Morning/Evening",
        "timing": "(6-10) AM / (2-5) PM"
    },
    {
        "id": 4,
        "activity": "Tharu Cultural Program",
        "prices": {"domestic": 200, "SAARC": 300, "tourist": 300},
        "schedule": "Evening",
        "timing": "(7-8) PM"
    },
    {
        "id": 5,
        "activity": "Jungle Walk",
        "prices": {"domestic": 5000, "SAARC": 10000, "tourist": 12500},
        "schedule": "Morning/Evening",
        "timing": "(6-10) AM / (2-5) PM"
    },
    {
        "id": 6,
        "activity": "Canoe Safari",
        "prices": {"domestic": 500, "SAARC": 600, "tourist": 700},
        "schedule": "Morning/Evening",
        "timing": "(6-10) AM / (2-5) PM"
    },
    {
        "id": 7,
        "activity": "Tharu Museum",
        "prices": {"domestic": 200, "SAARC": 400, "tourist": 400},
        "schedule": "Morning/Evening",
        "timing": "10:00 AM - 5:00 PM"
    }
]

output_path = Path(r"C:\Users\DELL\Desktop\AI_part\ai_service\app\data\raw\activities.json")

# Write with clean UTF-8 encoding
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Created clean activities.json at {output_path}")
print(f"File size: {output_path.stat().st_size} bytes")