SHELL := /bin/bash

OWNER          	= da
TYPE_APP        = pipeline
SERVICE_NAME    = trigger-s3-sqs
ENV             ?= prod

PROJECT_NAME    		= ${OWNER}-${TYPE_APP}-${SERVICE_NAME}-${ENV}

STACK ?= stack-${PROJECT_NAME}
REGION ?= us-east-2
PROFILE ?= default

# Source buckets (parametrize these with actual bucket names)
S3_SOURCE_BUCKET_1 ?= s3-bucket-prod-lake-zero
S3_DATA_LAKE_BUCKET ?= s3-bucket-prod-lake-analytics

PARAMS ?= \
  ParameterKey=ProjectName,ParameterValue=${PROJECT_NAME} \
  ParameterKey=Stage,ParameterValue=${ENV} \
  ParameterKey=Region,ParameterValue=$(REGION) \
  ParameterKey=CreateBucket,ParameterValue=false \
  ParameterKey=S3SourceBucket1,ParameterValue=$(S3_SOURCE_BUCKET_1) \
  ParameterKey=S3DataLakeBucket,ParameterValue=$(S3_DATA_LAKE_BUCKET) \
  ParameterKey=S3PrefixFilter,ParameterValue= \
  ParameterKey=SqsMaxReceiveCount,ParameterValue=3 \
  ParameterKey=LambdaMemorySize,ParameterValue=10240 \
  ParameterKey=LambdaTimeoutSeconds,ParameterValue=60 \
  ParameterKey=LambdaConcurrencyLimit,ParameterValue=1 \
  ParameterKey=SqsBatchSize,ParameterValue=5 \
  ParameterKey=SqsBatchWindowSeconds,ParameterValue=2 \
  ParameterKey=SqsVisibilityTimeoutSeconds,ParameterValue=180 \
  ParameterKey=LogLevel,ParameterValue=INFO \
  ParameterKey=EnableXRayTracing,ParameterValue=true \
  ParameterKey=KmsKeyArn,ParameterValue= \
  ParameterKey=CreateSnsAlarmTopic,ParameterValue=false \
  ParameterKey=TagProject,ParameterValue=Llamabara \
  ParameterKey=TagApplication,ParameterValue=head-to-head

define DCMD
docker compose run --rm -e AWS_PROFILE=$(PROFILE) -e AWS_REGION=$(REGION) sam "$(1)"
endef

build-docker:
	$(call DCMD,sam build)

deploy-docker: build-docker
	$(call DCMD,sam deploy --stack-name $(STACK) --region $(REGION) --capabilities CAPABILITY_NAMED_IAM CAPABILITY_IAM \
	  --no-fail-on-empty-changeset --parameter-overrides $(PARAMS) --confirm-changeset --resolve-s3)

# Testing targets (using Docker)
rebuild-test-image:
	docker compose build sam

test-unit:
	$(call DCMD,cd tests && python -m pytest test_app.py -v)

test-integration:
	$(call DCMD,cd tests && python -m pytest test_integration.py -v)

test-e2e:
	$(call DCMD,cd tests && python -m pytest test_e2e.py -v -s --tb=short)

test-all:
	$(call DCMD,cd tests && python -m pytest -v)

test-shell:
	$(call DCMD,bash)

# Stack management
delete-stack:
	@echo "⚠️  WARNING: This will permanently delete the stack '$(STACK)' and all its resources!"
	@echo "Stack: $(STACK)"
	@echo "Region: $(REGION)"
	@echo "Profile: $(PROFILE)"
	@ $(call DCMD,sam delete --stack-name $(STACK) --region $(REGION) --no-prompts)
  

delete-stack-force:
	@echo "🗑️  Force deleting stack $(STACK) without confirmation..."
	$(call DCMD,sam delete --stack-name $(STACK) --region $(REGION) --no-prompts)

# Help target
help:
	@echo "Available targets:"
	@echo "  build-docker       - Build SAM application in Docker"
	@echo "  deploy-docker      - Deploy stack to AWS"
	@echo "  delete-stack       - Delete stack with confirmation"
	@echo "  delete-stack-force - Delete stack without confirmation"
	@echo "  test-unit          - Run unit tests"
	@echo "  test-integration   - Run integration tests"
	@echo "  test-e2e          - Run end-to-end tests"
	@echo "  test-all          - Run all tests"
	@echo "  test-shell        - Open interactive shell"
	@echo ""
	@echo "Variables:"
	@echo "  STACK=$(STACK)"
	@echo "  REGION=$(REGION)"
	@echo "  PROFILE=$(PROFILE)"

.PHONY: build-docker deploy-docker delete-stack delete-stack-force help test-unit test-integration test-e2e test-all test-shell

