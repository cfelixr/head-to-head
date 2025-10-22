# S3 → SQS → Lambda Automation Pipeline

A serverless automation pipeline that processes S3 object creation events using AWS SAM, SQS queues, and Lambda functions with guaranteed sequential processing.

## 🏗️ Architecture

```
S3 Bucket → SQS Queue → Lambda Function
    ↓           ↓            ↓
Event Notification → DLQ → CloudWatch Logs
                     ↓
              CloudWatch Alarms
```

### Components

- **S3 Bucket**: Triggers events on object creation
- **SQS Standard Queue**: Receives S3 notifications, enables batching
- **Dead Letter Queue (DLQ)**: Handles failed message processing
- **Lambda Function**: Processes messages with concurrency limit of 1
- **CloudWatch Alarms**: Monitors DLQ for failed messages
- **X-Ray Tracing**: Optional distributed tracing

## ✨ Key Features

- **Sequential Processing**: `LambdaConcurrencyLimit: 1` ensures only one Lambda executes at a time
- **Fault Tolerance**: DLQ with configurable retry attempts
- **Batch Processing**: Configurable batch size and window
- **Monitoring**: CloudWatch alarms and optional SNS notifications
- **Security**: Optional KMS encryption for SQS queues
- **Observability**: X-Ray tracing support

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- Existing S3 bucket (or set `CreateBucket=true`)

### Deployment

1. **Clone and navigate to project**:
   ```bash
   cd automation-trigger-s3
   ```

2. **Configure parameters** in `Makefile`:
   ```makefile
   S3_BUCKET_RECEPTION = your-bucket-name
   REGION = us-east-2
   PROFILE = your-aws-profile
   ```

3. **Deploy the stack**:
   ```bash
   make deploy-docker
   ```

4. **Configure S3 Event Notifications**:
   - Go to AWS S3 Console
   - Select your bucket → Properties → Event notifications
   - Create notification pointing to the deployed SQS queue
   - Select event types: `s3:ObjectCreated:*`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PROJECT_NAME` | Unique project identifier | `da-pipeline-trigger-s3-sqs-dev` |
| `REGION` | AWS region | `us-east-2` |
| `PROFILE` | AWS CLI profile | `default` |

### Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `LambdaConcurrencyLimit` | Max concurrent executions | `1` | 1-1000 |
| `LambdaMemorySize` | Memory allocation (MB) | `512` | 128-10240 |
| `LambdaTimeoutSeconds` | Function timeout | `30` | 1-900 |
| `SqsBatchSize` | Messages per invocation | `5` | 1-10 |
| `SqsBatchWindowSeconds` | Max wait for batch | `2` | 0-300 |
| `SqsMaxReceiveCount` | Retries before DLQ | `3` | 1-1000 |
| `LogLevel` | Logging verbosity | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |

## 🛠️ Development

### Available Commands

```bash
# Build and deployment
make build-docker          # Build SAM application
make deploy-docker          # Deploy to AWS
make delete-stack           # Delete stack (with confirmation)
make delete-stack-force     # Force delete without confirmation

# Testing
make test-unit             # Run unit tests
make test-integration      # Run integration tests
make test-e2e             # Run end-to-end tests
make test-all             # Run all tests

# Development
make test-shell           # Interactive shell
make help                 # Show all commands
```

### Project Structure

```
.
├── src/
│   ├── app.py              # Lambda function code
│   └── requirements.txt    # Python dependencies
├── tests/
│   ├── test_app.py         # Unit tests
│   ├── test_integration.py # Integration tests
│   └── test_e2e.py        # End-to-end tests
├── template.yaml          # SAM template
├── Makefile              # Build and deployment commands
├── docker-compose.yml    # Docker configuration
└── README.md            # This file
```

### Local Testing

```bash
# Run unit tests
make test-unit

# Run all tests with coverage
make test-all

# Interactive development shell
make test-shell
```

## 🔧 Customization

### Lambda Function

Modify `src/app.py` to implement your business logic:

```python
def lambda_handler(event, context):
    """Process SQS messages from S3 events"""
    
    for record in event['Records']:
        # Extract S3 event from SQS message
        s3_event = json.loads(record['body'])
        
        # Your processing logic here
        process_s3_object(s3_event)
    
    return {"batchItemFailures": []}
```

### Adding Dependencies

Update `src/requirements.txt`:

```txt
boto3>=1.26.0
requests>=2.28.0
your-package>=1.0.0
```

## 📊 Monitoring

### CloudWatch Metrics

- **Lambda Duration**: Function execution time
- **Lambda Errors**: Failed invocations
- **SQS Messages**: Queue depth and processing rate
- **DLQ Messages**: Failed message count (triggers alarms)

### Alarms

- **DLQ Has Messages**: Triggers when messages appear in dead letter queue
- **Lambda Errors**: High error rate detection
- **Lambda Duration**: Performance monitoring

### Logs

```bash
# View Lambda logs
aws logs tail /aws/lambda/da-pipeline-trigger-s3-sqs-dev-processor --follow

# View specific log group
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/da-pipeline"
```

## 🔐 Security

### IAM Permissions

The Lambda function has minimal required permissions:
- S3 read access to configured bucket
- SQS receive/delete messages
- CloudWatch Logs write access
- X-Ray tracing (if enabled)

### Encryption

- **SQS**: Optional KMS encryption via `KmsKeyArn` parameter
- **Lambda**: Environment variables encrypted at rest
- **CloudWatch**: Logs encrypted by default

## 🚨 Troubleshooting

### Common Issues

1. **S3 notifications not working**:
   - Verify SQS queue policy allows S3 to send messages
   - Check S3 event notification configuration
   - Ensure bucket and queue are in same region

2. **Lambda not processing messages**:
   - Check Lambda function logs in CloudWatch
   - Verify SQS event source mapping
   - Check concurrency limits and throttling

3. **Messages in DLQ**:
   - Review Lambda error logs
   - Check message format and processing logic
   - Verify timeout settings

### Debug Commands

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name stack-da-pipeline-trigger-s3-sqs-dev

# View SQS queue attributes
aws sqs get-queue-attributes --queue-url <queue-url> --attribute-names All

# Test Lambda function
aws lambda invoke --function-name da-pipeline-trigger-s3-sqs-dev-processor output.json
```

## 📈 Performance Tuning

### Concurrency Control

- **Sequential Processing**: `LambdaConcurrencyLimit: 1` (current setting)
- **Parallel Processing**: Increase limit based on downstream capacity
- **Cost Optimization**: Balance concurrency vs execution time

### Batch Configuration

- **High Throughput**: Increase `SqsBatchSize` to 10
- **Low Latency**: Decrease `SqsBatchWindowSeconds` to 0
- **Memory Usage**: Adjust `LambdaMemorySize` based on batch size

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run test suite: `make test-all`
5. Submit pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Check CloudWatch logs first
- Review troubleshooting section
- Create GitHub issue with logs and configuration details
