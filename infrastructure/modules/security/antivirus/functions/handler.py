import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

QUARANTINE_BUCKET = os.environ.get('QUARANTINE_BUCKET')
INFECTED_ACTION = os.environ.get('INFECTED_ACTION', 'QUARANTINE')

# Simple ClamAV-like signature check (in production, use real antivirus)
def scan_file(content):
    """Mock antivirus scan - replace with actual antivirus"""
    # Check for EICAR test virus or known signatures
    if b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE" in content:
        return True  # Virus detected
    return False

def handle(event, context):
    """Main handler for antivirus scanning"""
    
    for record in event.get('messages', []):
        bucket = record['details']['bucket_id']
        object_key = record['details']['object_id']
        
        logger.info(f"Scanning {bucket}/{object_key}")
        
        # Initialize S3 client
        s3 = boto3.client('s3', endpoint_url='https://storage.yandexcloud.net')
        
        try:
            # Download object
            response = s3.get_object(Bucket=bucket, Key=object_key)
            content = response['Body'].read()
            
            # Scan for viruses
            if scan_file(content):
                logger.warning(f"VIRUS DETECTED: {bucket}/{object_key}")
                
                if INFECTED_ACTION == 'QUARANTINE':
                    # Copy to quarantine
                    s3.copy_object(
                        Bucket=QUARANTINE_BUCKET,
                        Key=f"{bucket}/{object_key}",
                        CopySource={'Bucket': bucket, 'Key': object_key}
                    )
                    # Delete original
                    s3.delete_object(Bucket=bucket, Key=object_key)
                    logger.info(f"Quarantined {bucket}/{object_key}")
                    
                elif INFECTED_ACTION == 'DELETE':
                    s3.delete_object(Bucket=bucket, Key=object_key)
                    logger.info(f"Deleted infected {bucket}/{object_key}")
                    
            else:
                logger.info(f"Clean: {bucket}/{object_key}")
                
        except Exception as e:
            logger.error(f"Error scanning {bucket}/{object_key}: {e}")
            raise
    
    return {"statusCode": 200, "body": "Scan complete"}