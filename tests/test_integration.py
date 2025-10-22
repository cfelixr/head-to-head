import json
import boto3
import pytest
from moto import mock_sqs, mock_s3, mock_events
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import lambda_handler


@mock_sqs
@mock_s3
@mock_events
class TestIntegration:
    """Integration tests for the complete flow"""
    
    def setup_method(self, method):
        """Setup AWS resources for testing"""
        # Create SQS queue
        self.sqs = boto3.client('sqs', region_name='us-east-2')
        self.queue_url = self.sqs.create_queue(
            QueueName='test-queue.fifo',
            Attributes={
                'FifoQueue': 'true',
                'ContentBasedDeduplication': 'true'
            }
        )['QueueUrl']
        
        # Create S3 bucket
        self.s3 = boto3.client('s3', region_name='us-east-2')
        self.bucket_name = 'test-bucket'
        self.s3.create_bucket(
            Bucket=self.bucket_name,
            CreateBucketConfiguration={'LocationConstraint': 'us-east-2'}
        )
    
    def create_sqs_event(self, s3_events):
        """Create SQS event with EventBridge messages"""
        records = []
        for i, s3_event in enumerate(s3_events):
            eventbridge_event = {
                "version": "0",
                "id": f"event-{i}",
                "detail-type": "Object Created",
                "source": "aws.s3",
                "time": "2023-01-01T00:00:00Z",
                "detail": s3_event
            }
            
            records.append({
                "messageId": f"msg-{i}",
                "receiptHandle": f"handle-{i}",
                "body": json.dumps(eventbridge_event),
                "attributes": {
                    "ApproximateReceiveCount": "1",
                    "SentTimestamp": "1640995200000"
                },
                "messageAttributes": {},
                "md5OfBody": "test-md5",
                "eventSource": "aws:sqs",
                "eventSourceARN": f"arn:aws:sqs:us-east-2:123456789012:test-queue.fifo",
                "awsRegion": "us-east-2"
            })
        
        return {"Records": records}
    
    def test_successful_batch_processing(self):
        """Test successful processing of multiple S3 events"""
        s3_events = [
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "file1.txt", "size": 1024, "etag": "abc123"}
            },
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "file2.txt", "size": 2048, "etag": "def456"}
            }
        ]
        
        sqs_event = self.create_sqs_event(s3_events)
        
        with patch('app.logger') as mock_logger:
            result = lambda_handler(sqs_event, None)
            
            assert result["batchItemFailures"] == []
            assert mock_logger.info.call_count >= 2  # At least one per processed object
    
    def test_partial_batch_failure(self):
        """Test partial failure handling in batch"""
        s3_events = [
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "success.txt", "size": 1024, "etag": "abc123"}
            },
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "fail.txt", "size": 2048, "etag": "def456"}
            }
        ]
        
        sqs_event = self.create_sqs_event(s3_events)
        
        result = lambda_handler(sqs_event, None)
        
        # Should have one failure (the "fail.txt" file)
        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "msg-1"
    
    def test_prefix_filtering(self):
        """Test S3 prefix filtering functionality"""
        s3_events = [
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "allowed/file.txt", "size": 1024, "etag": "abc123"}
            },
            {
                "bucket": {"name": self.bucket_name},
                "object": {"key": "blocked/file.txt", "size": 2048, "etag": "def456"}
            }
        ]
        
        sqs_event = self.create_sqs_event(s3_events)
        
        # Mock environment variable for prefix filtering
        with patch.dict(os.environ, {'S3_PREFIX_FILTER': 'allowed/'}):
            # Need to reload the app module to pick up the new environment variable
            import importlib
            import app
            importlib.reload(app)
            
            with patch('app.logger') as mock_logger:
                result = app.lambda_handler(sqs_event, None)
                
                assert result["batchItemFailures"] == []
                # Check that one was skipped due to prefix
                skip_calls = [call for call in mock_logger.info.call_args_list 
                             if 'skipped_by_prefix' in str(call)]
                assert len(skip_calls) == 1
    
    def test_empty_batch(self):
        """Test handling of empty SQS batch"""
        event = {"Records": []}
        
        result = lambda_handler(event, None)
        
        assert result["batchItemFailures"] == []
    
    def test_malformed_eventbridge_event(self):
        """Test handling of malformed EventBridge events"""
        records = [{
            "messageId": "malformed-msg",
            "body": "invalid-json-content",
            "attributes": {},
            "messageAttributes": {}
        }]
        
        event = {"Records": records}
        
        result = lambda_handler(event, None)
        
        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "malformed-msg"


if __name__ == "__main__":
    pytest.main([__file__])
