# ------------------------------------------------------------------------------
# Required Variables
# ------------------------------------------------------------------------------

variable "folder_id" {
  description = "Yandex Cloud folder ID where resources will be created"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., prod, staging, dev)"
  type        = string
}

# ------------------------------------------------------------------------------
# Bucket Configuration
# ------------------------------------------------------------------------------

variable "create_bucket" {
  description = "Whether to create a new bucket or use existing one"
  type        = bool
  default     = true
}

variable "bucket_name" {
  description = "Name of the S3 bucket for shared configuration"
  type        = string
  
  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{2,62}$", var.bucket_name))
    error_message = "Bucket name must be 3-63 characters, lowercase letters, numbers, hyphens, and periods only."
  }
}

variable "existing_bucket_name" {
  description = "Name of existing bucket to use (if create_bucket = false)"
  type        = string
  default     = ""
}

variable "bucket_acl" {
  description = "ACL for the bucket (private, public-read, etc.)"
  type        = string
  default     = "private"
  
  validation {
    condition     = contains(["private", "public-read", "public-read-write", "authenticated-read"], var.bucket_acl)
    error_message = "ACL must be one of: private, public-read, public-read-write, authenticated-read."
  }
}

variable "bucket_max_size_gb" {
  description = "Maximum size of the bucket in GB"
  type        = number
  default     = 10
}

# ------------------------------------------------------------------------------
# Versioning and Lifecycle
# ------------------------------------------------------------------------------

variable "enable_versioning" {
  description = "Enable versioning on the bucket"
  type        = bool
  default     = true
}

variable "enable_lifecycle_rules" {
  description = "Enable lifecycle rules for cleanup"
  type        = bool
  default     = true
}

variable "keep_versions_days" {
  description = "Number of days to keep old versions of config files"
  type        = number
  default     = 90
}

# ------------------------------------------------------------------------------
# Encryption
# ------------------------------------------------------------------------------

variable "sse_algorithm" {
  description = "Server-side encryption algorithm"
  type        = string
  default     = "AES256"
  
  validation {
    condition     = contains(["AES256", "aws:kms"], var.sse_algorithm)
    error_message = "SSE algorithm must be either AES256 or aws:kms."
  }
}

variable "kms_key_id" {
  description = "KMS key ID for encryption (if using aws:kms)"
  type        = string
  default     = null
}

# ------------------------------------------------------------------------------
# Initial Configuration
# ------------------------------------------------------------------------------

variable "create_initial_config" {
  description = "Whether to upload an initial configuration file"
  type        = bool
  default     = false
}

variable "config_file_key" {
  description = "Key (path) for the configuration file in the bucket"
  type        = string
  default     = "infrastructure-config.json"
}

variable "initial_config_content" {
  description = "Initial configuration content (will be JSON encoded)"
  type        = any
  default     = {}
}

# ------------------------------------------------------------------------------
# Access Control
# ------------------------------------------------------------------------------

variable "apply_bucket_policy" {
  description = "Whether to apply bucket policy for cross-repo access"
  type        = bool
  default     = false
}

variable "read_only_service_accounts" {
  description = "List of service account IDs that should have read-only access"
  type        = list(string)
  default     = []
}

variable "write_service_accounts" {
  description = "List of service account IDs that should have write access"
  type        = list(string)
  default     = []
}

# ------------------------------------------------------------------------------
# Tags and Metadata
# ------------------------------------------------------------------------------

variable "tags" {
  description = "Tags to apply to the bucket"
  type        = map(string)
  default = {
    managed_by = "terraform"
    purpose    = "shared-config"
  }
}