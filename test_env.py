"""
Test script to verify .env file is loading correctly
Run this BEFORE starting main.py
"""
from pathlib import Path
import os

print("\n" + "="*70)
print("🧪 TESTING ENVIRONMENT VARIABLE LOADING")
print("="*70)

# 1. Check current directory
current_dir = Path.cwd()
script_dir = Path(__file__).resolve().parent
print(f"\n📂 Current working directory: {current_dir}")
print(f"📂 Script directory: {script_dir}")

# 2. Look for .env file
env_file = script_dir / '.env'
env_file_cwd = current_dir / '.env'

print(f"\n🔍 Checking for .env file...")
print(f"   At script location: {env_file} → {'✅ EXISTS' if env_file.exists() else '❌ NOT FOUND'}")
print(f"   At current dir: {env_file_cwd} → {'✅ EXISTS' if env_file_cwd.exists() else '❌ NOT FOUND'}")

# 3. Try to read .env file directly
env_to_use = env_file if env_file.exists() else (env_file_cwd if env_file_cwd.exists() else None)

if env_to_use:
    print(f"\n📄 Reading .env file from: {env_to_use}")
    print("-"*70)
    try:
        with open(env_to_use, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Hide actual API key values
                    if '=' in line:
                        key, value = line.split('=', 1)
                        if 'KEY' in key.upper() or 'SECRET' in key.upper():
                            display_value = value[:10] + '...' if len(value) > 10 else value
                        else:
                            display_value = value
                        print(f"   Line {i}: {key}={display_value}")
                    else:
                        print(f"   Line {i}: {line} ⚠️ (Missing '=')")
                elif line.startswith('#'):
                    print(f"   Line {i}: {line} (comment)")
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
    print("-"*70)
else:
    print("\n❌ No .env file found!")
    print(f"💡 Please create .env file at: {script_dir / '.env'}")
    print("\nExpected format:")
    print("GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxx")
    print("REBUILD_INDEX=true")
    print("PORT=8000")
    print("HOST=0.0.0.0")

# 4. Try loading with python-dotenv
print("\n🔧 Testing python-dotenv loading...")
try:
    from dotenv import load_dotenv
    
    if env_to_use:
        result = load_dotenv(dotenv_path=env_to_use)
        print(f"   load_dotenv() returned: {result}")
    else:
        result = load_dotenv()
        print(f"   load_dotenv() (auto-search) returned: {result}")
    
    print("   ✅ python-dotenv imported successfully")
except ImportError:
    print("   ❌ python-dotenv not installed!")
    print("   💡 Run: pip install python-dotenv")

# 5. Check environment variables
print("\n🔍 Checking environment variables...")
keys_to_check = ["GOOGLE_API_KEY", "REBUILD_INDEX", "PORT", "HOST"]

for key in keys_to_check:
    value = os.getenv(key)
    if value:
        # Hide sensitive values
        if 'KEY' in key or 'SECRET' in key:
            display = value[:15] + '...' if len(value) > 15 else value
        else:
            display = value
        print(f"   ✅ {key} = {display}")
    else:
        print(f"   ❌ {key} = NOT FOUND")

# 6. Check for common issues
print("\n🔍 Checking for common issues...")

issues_found = []

if env_to_use:
    with open(env_to_use, 'r') as f:
        content = f.read()
        
        # Check for spaces around =
        if ' = ' in content:
            issues_found.append("⚠️  Spaces around '=' found (should be KEY=value, not KEY = value)")
        
        # Check for quotes
        if '="' in content or "='" in content:
            issues_found.append("⚠️  Quotes found around values (should be KEY=value, not KEY=\"value\")")
        
        # Check for Windows line endings
        if '\r\n' in content:
            issues_found.append("ℹ️  Windows line endings detected (usually OK)")
        
        # Check if any lines are too long (might be wrapped)
        for line in content.split('\n'):
            if len(line.strip()) > 200:
                issues_found.append("⚠️  Very long line detected (might be an issue)")

if issues_found:
    print("\n⚠️  Issues detected:")
    for issue in issues_found:
        print(f"   {issue}")
else:
    print("   ✅ No common issues detected")

# 7. Final verdict
print("\n" + "="*70)
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print("✅ SUCCESS: Environment variables loaded correctly!")
    print(f"✅ API Key detected (length: {len(api_key)} characters)")
    print("\n🚀 You can now run: python main.py")
else:
    print("❌ FAILED: GOOGLE_API_KEY not loaded")
    print("\n🔧 Troubleshooting steps:")
    print("   1. Ensure .env file exists in the same folder as main.py")
    print("   2. Check .env format (no spaces, no quotes):")
    print("      GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxx")
    print("   3. Verify file is named '.env' not '.env.txt'")
    print("   4. Try closing and reopening your terminal/IDE")
    print("   5. Run: pip install --upgrade python-dotenv")

print("="*70 + "\n")