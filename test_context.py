#!/usr/bin/env python3
"""
Test script for the context functionality in the emotion detection app
"""

import time
import sys
import os

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the context functions from our app
try:
    from app import (
        update_analysis_context, 
        get_context_summary, 
        reset_analysis_context,
        emotion_history,
        analysis_context
    )
    print("✅ Successfully imported context functions from app.py")
except ImportError as e:
    print(f"❌ Error importing from app.py: {e}")
    sys.exit(1)

def test_context_functionality():
    """Test the context tracking functionality"""
    print("\n🧪 Testing Context Functionality")
    print("=" * 50)
    
    # Reset context first
    reset_analysis_context()
    print("📄 Context reset")
    
    # Test 1: First analysis
    print("\n1️⃣ First Analysis:")
    update_analysis_context("Happy", 0.8, ["laptop", "coffee", "book"])
    print(f"   Context: {get_context_summary()}")
    print(f"   Emotion history: {[e['emotion'] for e in emotion_history]}")
    
    # Wait a moment and add another analysis
    time.sleep(1)
    
    # Test 2: Second analysis with same emotion
    print("\n2️⃣ Second Analysis (same emotion):")
    update_analysis_context("Happy", 0.9, ["laptop", "coffee", "plant"])
    print(f"   Context: {get_context_summary()}")
    print(f"   Emotion history: {[e['emotion'] for e in emotion_history]}")
    
    # Test 3: Third analysis with same emotion (should show consistency)
    print("\n3️⃣ Third Analysis (consistent emotion):")
    update_analysis_context("Happy", 0.85, ["laptop", "coffee", "book"])
    print(f"   Context: {get_context_summary()}")
    print(f"   Emotion trends: {analysis_context['emotion_trends']}")
    
    # Test 4: Change emotion
    print("\n4️⃣ Fourth Analysis (emotion change):")
    update_analysis_context("Neutral", 0.7, ["laptop", "phone"])
    print(f"   Context: {get_context_summary()}")
    print(f"   Emotion trends: {analysis_context['emotion_trends']}")
    
    # Test 5: Add more data to see object consistency
    print("\n5️⃣ Fifth Analysis (check object consistency):")
    update_analysis_context("Focused", 0.8, ["laptop", "book", "pen"])
    print(f"   Context: {get_context_summary()}")
    print(f"   Consistent objects: {analysis_context['object_consistency']}")
    
    print("\n✅ Context functionality test completed!")
    print(f"📊 Total interactions: {analysis_context['interaction_count']}")
    print(f"📈 Recent emotions: {analysis_context['previous_emotions']}")

if __name__ == "__main__":
    test_context_functionality()
