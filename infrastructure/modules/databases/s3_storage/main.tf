resource "yandex_storage_bucket" "s3_bucket" {
  # Storage bucket
  bucket    = var.s3_bucket_name
  folder_id = var.s3_yandex_cloud_folder_id

  default_storage_class = var.s3_default_storage_class
  
  # Versioning
  versioning {
    enabled = var.s3_versioning_enabled
  }

  # CORS configuration for cross-origin requests
  cors_rule {
    allowed_headers = ["*"]
    allowed_methods = ["GET", "HEAD", "PUT", "POST", "DELETE"]
    allowed_origins = ["*"]
    expose_headers  = ["ETag"]
    max_age_seconds = 3000
  }
  
  # Tags/Labels
  tags = var.s3_tags
}