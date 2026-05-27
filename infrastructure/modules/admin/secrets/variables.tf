variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for resource names"
  type        = string
  default     = ""
}

variable "secrets_manager_sa_id" {
  description = "Service account ID for secrets management"
  type        = string
}

# KMS Configuration
variable "kms_key_name" {
  description = "Name of the KMS key"
  type        = string
  default     = "default-kms-key"
}

variable "kms_description" {
  description = "Description of the KMS key"
  type        = string
  default     = "KMS key for encrypting secrets and data"
}

variable "kms_rotation_period" {
  description = "Rotation period for KMS key (e.g., '720h' for 30 days)"
  type        = string
  default     = "720h"
}

variable "kms_deletion_protection" {
  description = "Enable deletion protection for KMS key"
  type        = bool
  default     = true
}

# Lockbox Configuration
variable "lockbox_secret_name" {
  description = "Name of the Lockbox secret"
  type        = string
  default     = "default-secret"
}

variable "lockbox_description" {
  description = "Description of the Lockbox secret"
  type        = string
  default     = "Lockbox secret for storing application secrets"
}

variable "lockbox_deletion_protection" {
  description = "Enable deletion protection for Lockbox secret"
  type        = bool
  default     = true
}

variable "lockbox_version_payload" {
  description = "Key-value pairs for the initial secret version"
  type        = map(string)
  default     = {}
  sensitive   = true
}

# Labels
variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    managed-by = "terraform"
  }
}