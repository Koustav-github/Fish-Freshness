import time
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel

role = "arn:aws:iam::504509954111:role/SageMakerExecutionRole"
model_uri = "s3://amazon-sagemaker-504509954111-us-east-1-6bxrjj55zns7tz/fish-model/model.tar.gz"
endpoint_name = "fish-freshness-koustav-JU-km"

session = sagemaker.Session(boto_session=boto3.Session(region_name="us-east-1"))

model = PyTorchModel(
    model_data=model_uri,
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",
    sagemaker_session=session,
)

# Registers a new SageMaker Model resource: packages + uploads the updated
# inference.py and points at the (unchanged) model_data. Doesn't touch the
# live endpoint yet.
model.create(instance_type="ml.m5.large")
model_name = model.name
print("Created model:", model_name)

sm = boto3.client("sagemaker", region_name="us-east-1")

config_name = f"fish-freshness-config-{int(time.time())}"
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "ServerlessConfig": {
            "MemorySizeInMB": 3072,
            "MaxConcurrency": 5,
        },
    }],
)
print("Created endpoint config:", config_name)

sm.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=config_name)
print(f"Update initiated for endpoint '{endpoint_name}' -> config '{config_name}'")
