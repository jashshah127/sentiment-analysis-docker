"""
Test script for Dockerized Sentiment Analysis API
"""
import requests
import time
import sys

BASE_URL = "http://localhost:5000"

def wait_for_api(timeout=60):
    """Wait for API to be ready"""
    print("⏳ Waiting for Dockerized API to be ready...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ API is ready!")
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
            print(".", end="", flush=True)
    
    print("\n❌ API failed to start within timeout")
    return False

def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model loaded: {data['model_loaded']}")
            return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    return False

def test_positive_sentiment():
    """Test positive sentiment prediction"""
    print("\n🔍 Testing positive sentiment...")
    
    review = {
        "review": "This movie was absolutely amazing! Best film I've seen all year!"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=review)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Positive sentiment detected correctly")
            print(f"   Review: {data['original_text'][:50]}...")
            print(f"   Sentiment: {data['sentiment']}")
            print(f"   Score: {data['sentiment_score']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            return data['sentiment'] == 'positive'
    except Exception as e:
        print(f"❌ Positive test failed: {e}")
    return False

def test_negative_sentiment():
    """Test negative sentiment prediction"""
    print("\n🔍 Testing negative sentiment...")
    
    review = {
        "review": "Terrible movie. Complete waste of time and money. Very disappointing."
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=review)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Negative sentiment detected correctly")
            print(f"   Review: {data['original_text'][:50]}...")
            print(f"   Sentiment: {data['sentiment']}")
            print(f"   Score: {data['sentiment_score']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            return data['sentiment'] == 'negative'
    except Exception as e:
        print(f"❌ Negative test failed: {e}")
    return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model: {data['model_type']}")
            print(f"   Test Accuracy: {data['test_accuracy']:.4f}")
            print(f"   ROC-AUC: {data['roc_auc_score']:.4f}")
            print(f"   Vocabulary Size: {data['vocabulary_size']}")
            return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    return False

def test_error_handling():
    """Test API error handling"""
    print("\n🔍 Testing error handling...")
    
    # Missing review field
    invalid_data = {"text": "This is wrong field name"}
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
        if response.status_code == 400:
            print("✅ Error handling working - Invalid data rejected")
            return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    return False

def run_performance_test():
    """Run performance test"""
    print("\n🔍 Running performance test...")
    
    test_review = {
        "review": "This is a great movie with excellent acting!"
    }
    
    # Warm up
    requests.post(f"{BASE_URL}/predict", json=test_review)
    
    # Time multiple requests
    num_requests = 10
    start_time = time.time()
    
    for _ in range(num_requests):
        response = requests.post(f"{BASE_URL}/predict", json=test_review)
        if response.status_code != 200:
            print(f"❌ Performance test failed")
            return False
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_requests
    
    print(f"✅ Performance test completed")
    print(f"   {num_requests} requests in {total_time:.2f}s")
    print(f"   Average response time: {avg_time*1000:.2f}ms")
    print(f"   Requests per second: {num_requests/total_time:.2f}")
    
    return True

def main():
    """Run all Docker API tests"""
    print("🐳 Docker Sentiment Analysis API Tests")
    print("=" * 60)
    
    # Wait for API
    if not wait_for_api():
        print("\n💡 Make sure Docker container is running:")
        print("   docker-compose up")
        sys.exit(1)
    
    # Run tests
    tests = [
        test_health,
        test_positive_sentiment,
        test_negative_sentiment,
        test_model_info,
        test_error_handling,
        run_performance_test
    ]
    
    passed = sum(1 for test in tests if test())
    total = len(tests)
    
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Docker tests passed!")
        print("\n✅ Your Dockerized Sentiment Analysis API is working perfectly!")
        print("\nDocker commands:")
        print("  Stop:    docker-compose down")
        print("  Restart: docker-compose up")
        print("  Logs:    docker-compose logs -f")
    else:
        print("⚠️  Some tests failed")
        sys.exit(1)

if __name__ == "__main__":
    main()