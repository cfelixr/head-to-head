import json
import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from app import lambda_handler, process_object, _extract_s3_from_eventbridge


class TestLambdaHandler:
    """Unit tests for the main Lambda handler"""
    
    def test_lambda_handler_success(self):
        """Test successful processing of SQS batch"""
        event = {
            "Records": [
                {
                    "messageId": "test-msg-1",
                    "body": json.dumps({
                        "version": "0",
                        "id": "event-1",
                        "detail-type": "Object Created",
                        "source": "aws.s3",
                        "time": "2023-01-01T00:00:00Z",
                        "detail": {
                            "bucket": {"name": "test-bucket"},
                            "object": {"key": "test-file.txt", "size": 1024, "etag": "abc123"}
                        }
                    })
                }
            ]
        }
        
        with patch('app.process_object') as mock_process:
            result = lambda_handler(event, None)
            
            assert result == {"batchItemFailures": []}
            mock_process.assert_called_once()
    
    def test_lambda_handler_partial_failure(self):
        """Test partial batch failure handling"""
        event = {
            "Records": [
                {
                    "messageId": "success-msg",
                    "body": json.dumps({
                        "detail": {
                            "bucket": {"name": "test-bucket"},
                            "object": {"key": "success.txt", "size": 1024}
                        }
                    })
                },
                {
                    "messageId": "fail-msg", 
                    "body": json.dumps({
                        "detail": {
                            "bucket": {"name": "test-bucket"},
                            "object": {"key": "fail.txt", "size": 1024}
                        }
                    })
                }
            ]
        }
        
        def mock_process_side_effect(bucket, key, metadata):
            if "fail" in key:
                raise RuntimeError("Simulated failure")
        
        with patch('app.process_object', side_effect=mock_process_side_effect):
            result = lambda_handler(event, None)
            
            assert len(result["batchItemFailures"]) == 1
            assert result["batchItemFailures"][0]["itemIdentifier"] == "fail-msg"
    
    def test_lambda_handler_malformed_body(self):
        """Test handling of malformed JSON in message body"""
        event = {
            "Records": [
                {
                    "messageId": "malformed-msg",
                    "body": "invalid-json"
                }
            ]
        }
        
        result = lambda_handler(event, None)
        
        assert len(result["batchItemFailures"]) == 1
        assert result["batchItemFailures"][0]["itemIdentifier"] == "malformed-msg"


class TestProcessObject:
    """Unit tests for the process_object function"""
    
    def test_process_object_success(self):
        """Test successful object processing"""
        with patch('app.logger') as mock_logger:
            process_object("test-bucket", "success.txt", {"size": 1024})
            mock_logger.info.assert_called_once()
    
    def test_process_object_failure(self):
        """Test object processing failure simulation"""
        with pytest.raises(RuntimeError, match="Simulated processing failure"):
            process_object("test-bucket", "fail.txt", {"size": 1024})


class TestS3EventExtraction:
    """Unit tests for EventBridge event parsing"""
    
    def test_extract_s3_from_eventbridge(self):
        """Test S3 metadata extraction from EventBridge event"""
        event = {
            "version": "0",
            "id": "test-event-id",
            "time": "2023-01-01T00:00:00Z",
            "detail": {
                "bucket": {"name": "test-bucket"},
                "object": {"key": "test-file.txt", "size": 2048, "etag": "def456"}
            }
        }
        
        result = _extract_s3_from_eventbridge(event)
        
        assert result["bucket"] == "test-bucket"
        assert result["key"] == "test-file.txt"
        assert result["size"] == 2048
        assert result["eTag"] == "def456"
        assert result["eventTime"] == "2023-01-01T00:00:00Z"
        assert result["id"] == "test-event-id"
    
    def test_extract_s3_missing_fields(self):
        """Test handling of missing fields in EventBridge event"""
        event = {"detail": {}}
        
        result = _extract_s3_from_eventbridge(event)
        
        assert result["bucket"] is None
        assert result["key"] is None
        assert result["size"] is None


if __name__ == "__main__":
    pytest.main([__file__])
