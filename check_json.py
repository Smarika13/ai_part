# Save this as check_json.py in your project root
from pathlib import Path

activity_file = Path(r"C:\Users\DELL\Desktop\AI_part\ai_service\app\data\raw\activities.json")

print("=== FILE DIAGNOSIS ===")
print(f"File exists: {activity_file.exists()}")
print(f"File size: {activity_file.stat().st_size if activity_file.exists() else 'N/A'} bytes")

if activity_file.exists():
    # Read as bytes to see raw content
    with open(activity_file, 'rb') as f:
        raw_bytes = f.read()
        print(f"\nFirst 50 bytes (hex): {raw_bytes[:50].hex()}")
        print(f"First 50 bytes (repr): {repr(raw_bytes[:50])}")
    
    # Try different encodings
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
        try:
            with open(activity_file, 'r', encoding=encoding) as f:
                content = f.read()
                print(f"\n✅ {encoding}: {len(content)} characters")
                print(f"   First 100 chars: {repr(content[:100])}")
        except Exception as e:
            print(f"\n❌ {encoding}: {e}")