import os
import time
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

# Setup session and role
role = "arn:aws:iam::504509954111:role/SageMakerExecutionRole"
model_uri = "s3://amazon-sagemaker-504509954111-us-east-1-6bxrjj55zns7tz/fish-model/model.tar.gz"
endpoint_name = "fish-freshness-koustav-JU-km"  # must match SAGEMAKER_ENDPOINT_NAME in backend/main.py
region = "us-east-1"
memory_size_mb = 3072  # this account's current serverless memory quota ceiling

session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

model = PyTorchModel(
    model_data=model_uri,
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    # --- CHANGED SECTION: root cause fix ---
    # Without source_dir, the SDK only packages the single entry_point file
    # into code/ — requirements.txt silently never gets included, so none of
    # our pip dependencies (onnxruntime etc.) were ever actually installed in
    # the container. This is what every "Worker died" crash today actually
    # was, regardless of which framework we tried.
    source_dir=CODE_DIR,
    # -----------------------
    # --- CHANGED SECTION: startup timeout fix ---
    # TorchServe's default worker startup/response timeout is too short for
    # loading ~187MB across two ONNX models on constrained CPU — the worker
    # was being killed and retried (Fibonacci backoff) before it ever
    # finished loading, even though loading itself was no longer crashing.
    env={
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "300",
        "TS_DEFAULT_RESPONSE_TIMEOUT": "300",
    },
    # -----------------------
    sagemaker_session=session,
)

# Registers a new SageMaker Model resource: packages + uploads inference.py
# and requirements.txt, points at model_data. Doesn't touch any endpoint yet.
# (model.deploy()'s built-in update_endpoint=True has an SDK bug for
# serverless configs — it assumes an instance-based deploy — so we drive the
# endpoint config/update steps directly via boto3 below instead.)
model.create(instance_type="ml.m5.large")
model_name = model.name
print("Created model:", model_name)

sm = boto3.client("sagemaker", region_name=region)

config_name = f"fish-freshness-config-{int(time.time())}"
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "ServerlessConfig": {
            "MemorySizeInMB": memory_size_mb,
            "MaxConcurrency": 5,
        },
    }],
)
print("Created endpoint config:", config_name)

try:
    sm.describe_endpoint(EndpointName=endpoint_name)
    endpoint_exists = True
except sm.exceptions.ClientError:
    endpoint_exists = False

if endpoint_exists:
    sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"Update initiated for existing endpoint '{endpoint_name}' -> config '{config_name}'")
else:
    sm.create_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
    print(f"Created new endpoint '{endpoint_name}' -> config '{config_name}'")

print("Run this to poll status: aws sagemaker describe-endpoint --endpoint-name "
      f"{endpoint_name} --region {region} --query EndpointStatus --output text")
