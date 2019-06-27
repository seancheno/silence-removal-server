import boto3
from botocore.client import Config
import config as config

s3 = boto3.resource(
    's3',
    verify=False,
    aws_access_key_id=config.ACCESS_KEY_ID,
    aws_secret_access_key=config.ACCESS_SECRET_KEY,
    config=Config(signature_version='s3v4')
)

