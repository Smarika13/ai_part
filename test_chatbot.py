"""
Comprehensive Test Suite for Chitwan National Park Chatbot
Tests various question types to ensure the chatbot always provides answers
"""

from app.services.rag_service import RAGService
from dotenv import load_dotenv
import time

# Load environment
load_dotenv()

# Initialize RAG
print("🔧 Initializing RAG Service...")
rag = RAGService()
rag.initialize(rebuild_index=False)
print("✅ RAG Service Ready!\n")

# Test cases covering all aspects
TEST_CASES = {
    "Wildlife - Birds": [
        "List 5 birds found in Chitwan National Park",
        "Tell me about endangered bird species",
        "What birds can I see in Chitwan?",
        "Are there any kingfishers in the park?",
        "Tell me about the Bengal Florican",
        "What is the scientific name of the Great Hornbill?",
        "Which birds are critically endangered?",
    ],
    
    "Wildlife - Mammals": [
        "Tell me about the one-horned rhinoceros",
        "Are there tigers in Chitwan?",
        "What mammals live in Chitwan National Park?",
        "Tell me about Bengal Tiger",
        "What is the habitat of rhinos?",
        "List endangered mammals",
    ],
    
    "Wildlife - Reptiles": [
        "What reptiles can I see?",
        "Are there crocodiles in Chitwan?",
        "Tell me about gharials",
        "What snakes are found in the park?",
    ],
    
    "Park Rules & Regulations": [
        "What are the park rules?",
        "Can I bring plastic bags?",
        "Is feeding animals allowed?",
        "What is prohibited in the park?",
        "Can I bring a drone?",
        "What are the safety guidelines?",
        "Am I allowed to smoke?",
    ],
    
    "Visitor Information": [
        "What time does the park open?",
        "What are the entry fees?",
        "How much does it cost to enter?",
        "When is the best time to visit?",
        "What activities are available?",
        "How long is the jungle safari?",
        "Where can I buy tickets?",
        "Are there hotels in the park?",
    ],
    
    "General Questions": [
        "Tell me about Chitwan National Park",
        "What can I do in Chitwan?",
        "Is it safe to visit?",
        "What should I bring?",
        "Do I need a guide?",
        "How do I get to Chitwan?",
    ],
    
    "Edge Cases": [
        "Hello",
        "What?",
        "Tell me everything",
        "xyz123",  # Nonsense query
        "What is the capital of Nepal?",  # Out of scope
        "How many species are there?",
        "Compare birds and mammals",
    ]
}

def test_query(question, category):
    """Test a single query and return results"""
    try:
        result = rag.query(question)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # Check if answer is meaningful
        is_valid = (
            answer and 
            answer.strip() and 
            "I don't know" not in answer and
            "don't have" not in answer.lower() and
            len(answer) > 20  # At least 20 characters
        )
        
        return {
            "question": question,
            "category": category,
            "answer": answer,
            "sources": sources,
            "valid": is_valid,
            "answer_length": len(answer)
        }
    except Exception as e:
        return {
            "question": question,
            "category": category,
            "answer": f"ERROR: {str(e)}",
            "sources": [],
            "valid": False,
            "answer_length": 0
        }

def run_comprehensive_test():
    """Run all test cases and generate report"""
    print("="*80)
    print("🧪 RUNNING COMPREHENSIVE CHATBOT TESTS")
    print("="*80)
    print()
    
    all_results = []
    category_stats = {}
    
    total_tests = sum(len(questions) for questions in TEST_CASES.values())
    current_test = 0
    
    # Run all tests
    for category, questions in TEST_CASES.items():
        print(f"\n📂 Testing Category: {category}")
        print("-" * 60)
        
        category_results = []
        
        for question in questions:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Testing: {question}")
            
            result = test_query(question, category)
            category_results.append(result)
            all_results.append(result)
            
            # Print result
            if result["valid"]:
                print(f"  ✅ PASS - Answer length: {result['answer_length']} chars")
                print(f"  📚 Sources: {', '.join(result['sources']) if result['sources'] else 'None'}")
            else:
                print(f"  ❌ FAIL - {result['answer'][:100]}...")
            
            time.sleep(0.5)  # Avoid rate limits
        
        # Category statistics
        valid_count = sum(1 for r in category_results if r["valid"])
        category_stats[category] = {
            "total": len(category_results),
            "passed": valid_count,
            "failed": len(category_results) - valid_count,
            "pass_rate": (valid_count / len(category_results)) * 100
        }
    
    # Generate Report
    print("\n" + "="*80)
    print("📊 TEST RESULTS SUMMARY")
    print("="*80)
    
    # Overall stats
    total_valid = sum(1 for r in all_results if r["valid"])
    total_invalid = len(all_results) - total_valid
    overall_pass_rate = (total_valid / len(all_results)) * 100
    
    print(f"\n🎯 Overall Performance:")
    print(f"  Total Tests: {len(all_results)}")
    print(f"  ✅ Passed: {total_valid}")
    print(f"  ❌ Failed: {total_invalid}")
    print(f"  📈 Pass Rate: {overall_pass_rate:.1f}%")
    
    # Category breakdown
    print(f"\n📂 Category Breakdown:")
    for category, stats in category_stats.items():
        status = "✅" if stats["pass_rate"] >= 80 else "⚠️" if stats["pass_rate"] >= 60 else "❌"
        print(f"\n  {status} {category}:")
        print(f"     Passed: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")
    
    # Failed tests detail
    failed_tests = [r for r in all_results if not r["valid"]]
    if failed_tests:
        print(f"\n❌ Failed Tests Details:")
        print("-" * 80)
        for i, test in enumerate(failed_tests, 1):
            print(f"\n{i}. Category: {test['category']}")
            print(f"   Question: {test['question']}")
            print(f"   Answer: {test['answer'][:200]}...")
            print(f"   Sources: {test['sources']}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if overall_pass_rate >= 90:
        print("  ✅ Excellent! Your chatbot is working very well.")
    elif overall_pass_rate >= 70:
        print("  ⚠️  Good, but some improvements needed:")
        print("     - Check failed test categories")
        print("     - Ensure all data files are properly loaded")
        print("     - Consider improving prompt for edge cases")
    else:
        print("  ❌ Needs improvement:")
        print("     1. Rebuild the FAISS index: rag.initialize(rebuild_index=True)")
        print("     2. Verify all JSON and TXT files exist and are valid")
        print("     3. Check if custom prompt is properly added")
        print("     4. Review failed test details above")
    
    return all_results, category_stats

def quick_test():
    """Quick test with just a few key questions"""
    print("⚡ QUICK TEST (5 questions)")
    print("="*60)
    
    quick_questions = [
        "List 5 birds in Chitwan",
        "What are the entry fees?",
        "Can I feed animals?",
        "Tell me about rhinos",
        "What time does the park open?"
    ]
    
    results = []
    for i, q in enumerate(quick_questions, 1):
        print(f"\n{i}. {q}")
        result = test_query(q, "Quick Test")
        results.append(result)
        
        if result["valid"]:
            print(f"   ✅ {result['answer'][:100]}...")
        else:
            print(f"   ❌ {result['answer']}")
    
    passed = sum(1 for r in results if r["valid"])
    print(f"\n📊 Result: {passed}/{len(quick_questions)} passed")
    
    return results

def test_retrieval_only():
    """Test if retrieval is finding relevant documents"""
    print("\n🔍 TESTING RETRIEVAL (without LLM)")
    print("="*60)
    
    test_queries = [
        "birds in chitwan",
        "park entry fees",
        "rhinoceros",
        "prohibited activities"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        docs = rag.vector_db.similarity_search(query, k=3)
        print(f"Found {len(docs)} relevant documents:")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i}. {source}: {preview}...")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode
        quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "retrieval":
        # Test retrieval only
        test_retrieval_only()
    else:
        # Full comprehensive test
        run_comprehensive_test()
    
    print("\n" + "="*80)
    print("✅ Testing Complete!")
    print("="*80)