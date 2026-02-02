print("Test 1: Script is running")

try:
    print("Test 2: Importing sentence_transformers...")
    import sentence_transformers
    print("SUCCESS: sentence_transformers imported!")
except Exception as e:
    print(f"ERROR: {e}")

try:
    print("Test 3: Importing HuggingFaceEmbeddings...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("SUCCESS: HuggingFaceEmbeddings imported!")
except Exception as e:
    print(f"ERROR: {e}")

print("Test completed!")