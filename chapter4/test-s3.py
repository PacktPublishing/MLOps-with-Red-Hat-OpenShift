import boto3, os

# these are the env variable injected by Red Hat Data Science based on your Data Connection settings. 
# env variables injected
# AWS_S3_ENDPOINT=http://minio-ml-workshop.ml-workshop.svc.cluster.local
# AWS_DEFAULT_REGION=us-east-1
# AWS_SECRET_ACCESS_KEY=minio123
# AWS_S3_BUCKET=demo-project
# AWS_ACCESS_KEY_ID=minio
 
BUCKET = "demo-files" #os.getenv("AWS_S3_BUCKET")
HTTP = 'http://'
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGIONNAME = os.getenv("AWS_DEFAULT_REGION")
AWS_S3_ENDPOINT = os.getenv("AWS_S3_ENDPOINT")

 
FILENAME_ON_S3 = 'file.txt'
FILE_ON_DISK = 'file.txt'
 
file = open(f"{FILE_ON_DISK}", "w")
file.write("Hello There \n")
file.close()
 
s3 = boto3.resource('s3',endpoint_url = AWS_S3_ENDPOINT + ":9000",
                    aws_access_key_id=AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
 
s3.Bucket(BUCKET).upload_file(FILE_ON_DISK, FILENAME_ON_S3)
 
 
print('uploading complete')
 
