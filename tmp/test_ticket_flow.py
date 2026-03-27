import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.rag.agent_graph import _classify_intent_async, _save_memory, _load_memory

async def test():
    session_id = "test_session_123"
    
    # 1. Test standard RAG classification
    print("Test 1: Normal question")
    res = await _classify_intent_async("What is the leave policy?", session_id=session_id)
    print(f"Result: {res}")
    
    # 2. Simulate pending ticket
    print("\nTest 2: Simulating pending ticket and saying 'yes'")
    await _save_memory(session_id, {"ticket_pending": True})
    res = await _classify_intent_async("yes", session_id=session_id)
    print(f"Result: {res}")
    
    print("\nTest 3: Simulating pending ticket and saying 'do it'")
    res = await _classify_intent_async("do it", session_id=session_id)
    print(f"Result: {res}")

    print("\nTest 4: Simulating pending ticket and saying 'create ticket'")
    res = await _classify_intent_async("create ticket", session_id=session_id)
    print(f"Result: {res}")

    # Clean up
    await _save_memory(session_id, {})

if __name__ == "__main__":
    asyncio.run(test())
