import boto3
import os

BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGIONNAME = os.getenv("AWS_DEFAULT_REGION")
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")

FILENAME_ON_S3 = 'file.txt'
FILE_ON_DISK = 'file.txt'

file = open(f"{FILE_ON_DISK}", "w")
file.write("Hello There \n")
file.close()

s3 = boto3.resource('s3', endpoint_url=AWS_S3_ENDPOINT,
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                    region_name=REGIONNAME)

s3.Bucket(BUCKET).upload_file(FILE_ON_DISK, FILENAME_ON_S3)

print('uploading complete')
