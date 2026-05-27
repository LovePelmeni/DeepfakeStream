variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for antivirus resources"
  type        = string
  default     = "antivirus"
}

variable "scan_buckets" {
  description = "List of bucket names or IDs to enable antivirus scanning on"
  type        = list(string)
  default     = []
}

variable "scan_on_upload" {
  description = "Scan files immediately upon upload"
  type        = bool
  default     = true
}

variable "infected_files_action" {
  description = "Action for infected files: 'QUARANTINE', 'DELETE', or 'BLOCK'"
  type        = string
  default     = "QUARANTINE"
  
  validation {
    condition     = contains(["QUARANTINE", "DELETE", "BLOCK"], var.infected_files_action)
    error_message = "Infected files action must be QUARANTINE, DELETE, or BLOCK"
  }
}

# Function configuration
variable "function_memory_mb" {
  description = "Memory for antivirus scanning function in MB"
  type        = number
  default     = 256
}

variable "function_timeout_seconds" {
  description = "Execution timeout for scanning function"
  type        = number
  default     = 60
}

variable "function_runtime" {
  description = "Runtime for the cloud function"
  type        = string
  default     = "python312"
}

variable "function_entrypoint" {
  description = "Entrypoint for the cloud function"
  type        = string
  default     = "handler.handle"
}

variable "function_zip_path" {
  description = "Path to the function zip file (optional)"
  type        = string
  default     = null
}

# Trigger configuration
variable "batch_cutoff" {
  description = "Maximum duration (in seconds) to wait for message batch before invoking the function"
  type        = number
  default     = 10
}

variable "retry_attempts" {
  description = "Number of retry attempts for failed function invocations"
  type        = number
  default     = 3
}

variable "retry_interval" {
  description = "Interval in seconds between retry attempts"
  type        = number
  default     = 10
}

variable "quarantine_retention_days" {
  description = "Days to keep quarantined files"
  type        = number
  default     = 90
}

# ✅ KMS Encryption Variables
variable "kms_key_id" {
  description = "ID of the KMS key to use for bucket encryption"
  type        = string
  default     = null
}

variable "kms_master_key_id" {
  description = "Master KMS key ID for bucket encryption (alias for kms_key_id)"
  type        = string
  default     = null
}

variable "kms_master_key_version" {
  description = "Version of the KMS key to use (optional, uses default if not specified)"
  type        = string
  default     = null
}

variable "use_kms_encryption" {
  description = "Whether to use KMS encryption instead of AES256"
  type        = bool
  default     = false
}

# Labels
variable "labels" {
  description = "Labels for antivirus resources"
  type        = map(string)
  default     = {}
}