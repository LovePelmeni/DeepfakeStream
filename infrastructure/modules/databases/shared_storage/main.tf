# ------------------------------------------------------------------------------
# S3 Bucket for Shared Configuration Storage
# ------------------------------------------------------------------------------

# Use existing bucket or create new one based on variable
locals {
  bucket_name = var.create_bucket ? var.bucket_name : var.existing_bucket_name
}

# Create S3 bucket if specified
resource "yandex_storage_bucket" "config_bucket" {
  
  count = var.create_bucket ? 1 : 0
  bucket     = var.bucket_name
  acl        = var.bucket_acl
  folder_id  = var.folder_id
  max_size   = var.bucket_max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
  
  # Enable versioning to track changes
  versioning {
    enabled = var.enable_versioning
  }
  
  # Server-side encryption
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = var.sse_algorithm
      }
    }
  }
  
  # Lifecycle rules to clean up old config versions
  lifecycle_rule {
    id      = "cleanup-old-configs"
    enabled = var.enable_lifecycle_rules
    
    # Keep only last N versions of config files
    noncurrent_version_expiration {
      days = var.keep_versions_days
    }
  }
  
  tags = var.tags
}

# ------------------------------------------------------------------------------
# Upload initial configuration (optional)
# ------------------------------------------------------------------------------

# Create initial config file if content provided
resource "yandex_storage_object" "initial_config" {
  count = var.create_initial_config ? 1 : 0
  
  bucket = local.bucket_name
  key    = var.config_file_key
  content = jsonencode(var.initial_config_content)
  
  # Set content type
  content_type = "application/json"
  
  # Add metadata
  metadata = {
    "created-by" = "terraform"
    "environment" = var.environment
    "last-updated" = timestamp()
  }
}

# ------------------------------------------------------------------------------
# Bucket Policy for Cross-Repo Access
# ------------------------------------------------------------------------------

# Grant read access to application service accounts
resource "yandex_storage_bucket_policy" "config_bucket_policy" {
  count = var.create_bucket && var.apply_bucket_policy ? 1 : 0
  
  bucket = yandex_storage_bucket.config_bucket[0].bucket
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          "AWS" : var.read_only_service_accounts
        }
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      },
      {
        Effect = "Allow"
        Principal = {
          "AWS" : var.write_service_accounts
        }
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      }
    ]
  })
}

# ------------------------------------------------------------------------------
# Output the bucket info for other modules
# ------------------------------------------------------------------------------

locals {
  bucket_id   = var.create_bucket ? yandex_storage_bucket.config_bucket[0].id : var.existing_bucket_name
  bucket_name = var.create_bucket ? yandex_storage_bucket.config_bucket[0].bucket : var.existing_bucket_name
}