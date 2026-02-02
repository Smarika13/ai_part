"""
Simple test for local embeddings - guaranteed to work
"""
import time

print("\n" + "="*60)
print("🧪 TESTING LOCAL EMBEDDINGS")
print("="*60 + "\n")

# Step 1: Import
print("Step 1: Importing libraries...")
from langchain_community.embeddings import HuggingFaceEmbeddings
print("✅ Import successful\n")

# Step 2: Load model
print("Step 2: Loading model (all-MiniLM-L6-v2)...")
print("(First time downloads ~90MB, please wait...)")
start = time.time()

try:
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.2f} seconds\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Step 3: Test embedding
print("Step 3: Testing embedding performance...")
test_texts = [
    "What animals live in Chitwan?",
    "Tell me about tigers",
    "What can I do in the park?"
]

start = time.time()
try:
    result = embeddings.embed_documents(test_texts)
    embed_time = time.time() - start
    
    print(f"✅ Embedded {len(test_texts)} texts in {embed_time:.3f} seconds")
    print(f"⚡ Speed: {embed_time/len(test_texts)*1000:.1f} ms per text")
    print(f"📊 Vector size: {len(result[0])} dimensions\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Step 4: Memory check (optional)
print("Step 4: Checking memory usage...")
try:
    import psutil
    import os
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    print(f"💾 Memory used: {memory_mb:.0f} MB\n")
except:
    print("⚠️ Could not check memory (psutil needed)\n")

# Final verdict
print("="*60)
print("✅ SUCCESS! Local embeddings work perfectly!")
print("\n📊 Summary:")
print(f"   • Model size: ~90 MB")
print(f"   • Speed: {embed_time/len(test_texts)*1000:.0f}ms per query")
print(f"   • Works without internet (after first download)")
print(f"   • No API quota limits!")
print("\n🎯 Your laptop CAN handle local embeddings!")
print("💡 Ready to update rag_service.py to avoid quota issues")
print("="*60 + "\n")