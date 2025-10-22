import pytest
import os
import sys

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture(scope="session")
def aws_credentials():
    """Mock AWS credentials for testing"""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-2'

@pytest.fixture
def lambda_context():
    """Mock Lambda context object"""
    class MockContext:
        def __init__(self):
            self.function_name = "test-function"
            self.function_version = "$LATEST"
            self.invoked_function_arn = "arn:aws:lambda:us-east-2:123456789012:function:test-function"
            self.memory_limit_in_mb = "512"
            self.remaining_time_in_millis = lambda: 30000
            self.log_group_name = "/aws/lambda/test-function"
            self.log_stream_name = "2023/01/01/[$LATEST]test123"
            self.aws_request_id = "test-request-id"
    
    return MockContext()

@pytest.fixture
def sample_s3_event():
    """Sample S3 event data for testing"""
    return {
        "bucket": {"name": "test-bucket"},
        "object": {
            "key": "test-file.txt",
            "size": 1024,
            "etag": "abc123def456",
        }
    }

@pytest.fixture
def sample_eventbridge_event(sample_s3_event):
    """Sample EventBridge event wrapping S3 data"""
    return {
        "version": "0",
        "id": "test-event-id",
        "detail-type": "Object Created",
        "source": "aws.s3",
        "account": "123456789012",
        "time": "2023-01-01T00:00:00Z",
        "region": "us-east-2",
        "resources": [f"arn:aws:s3:::{sample_s3_event['bucket']['name']}"],
        "detail": sample_s3_event
    }

@pytest.fixture
def sample_sqs_record(sample_eventbridge_event):
    """Sample SQS record containing EventBridge event"""
    import json
    return {
        "messageId": "test-message-id",
        "receiptHandle": "test-receipt-handle",
        "body": json.dumps(sample_eventbridge_event),
        "attributes": {
            "ApproximateReceiveCount": "1",
            "SentTimestamp": "1640995200000",
            "MessageGroupId": "default-group",
            "MessageDeduplicationId": "test-dedup-id"
        },
        "messageAttributes": {},
        "md5OfBody": "test-md5",
        "eventSource": "aws:sqs",
        "eventSourceARN": "arn:aws:sqs:us-east-2:123456789012:main-events.fifo",
        "awsRegion": "us-east-2"
    }
