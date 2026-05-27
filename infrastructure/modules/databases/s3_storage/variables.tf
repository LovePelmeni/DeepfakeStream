# S3 Bucket Variables

variable "s3_bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
}

variable "s3_yandex_cloud_id" {
  description = "Yandex Cloud ID for S3 bucket"
  type        = string
}

variable "s3_yandex_cloud_folder_id" {
  description = "Yandex Cloud Folder ID for S3 bucket environment"
  type        = string
}

variable "s3_default_storage_class" {
  description = "Default storage class: STANDARD, COLD, or ICE"
  type        = string
  default     = "STANDARD"
  
  validation {
    condition     = contains(["STANDARD", "COLD", "ICE"], var.s3_default_storage_class)
    error_message = "Storage class must be STANDARD, COLD, or ICE"
  }
}

variable "s3_versioning_enabled" {
  description = "Enable object versioning for S3 bucket"
  type        = bool
  default     = false
}

variable "s3_tags" {
  description = "Labels/tags for the S3 bucket"
  type        = map(string)
  default     = {}
}