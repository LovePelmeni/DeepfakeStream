# ------------------------------------------------------------------------------
# Bucket Information
# ------------------------------------------------------------------------------

output "bucket_id" {
  description = "ID of the S3 bucket"
  value       = local.bucket_id
}

output "bucket_name" {
  description = "Name of the S3 bucket"
  value       = local.bucket_name
}

output "bucket_domain" {
  description = "Domain name of the bucket"
  value       = var.create_bucket ? yandex_storage_bucket.config_bucket[0].bucket_domain_name : "${var.existing_bucket_name}.storage.yandexcloud.net"
}

output "bucket_endpoint" {
  description = "Endpoint URL for the bucket"
  value       = "https://${local.bucket_name}.storage.yandexcloud.net"
}

# ------------------------------------------------------------------------------
# Configuration File Information
# ------------------------------------------------------------------------------

output "config_file_key" {
  description = "Key of the configuration file in the bucket"
  value       = var.create_initial_config ? yandex_storage_object.initial_config[0].key : var.config_file_key
}

output "config_file_url" {
  description = "Full URL to the configuration file"
  value       = var.create_initial_config ? "https://${local.bucket_name}.storage.yandexcloud.net/${yandex_storage_object.initial_config[0].key}" : null
}

output "config_file_version_id" {
  description = "Version ID of the initial configuration file"
  value       = var.create_initial_config ? yandex_storage_object.initial_config[0].version_id : null
}

# ------------------------------------------------------------------------------
# Access Information
# ------------------------------------------------------------------------------

output "bucket_reader_iam_role" {
  description = "IAM role name for read access"
  value       = "arn:aws:s3:::${local.bucket_name}/reader"
}

output "bucket_writer_iam_role" {
  description = "IAM role name for write access"
  value       = "arn:aws:s3:::${local.bucket_name}/writer"
}

# ------------------------------------------------------------------------------
# Utility Commands
# ------------------------------------------------------------------------------

output "aws_cli_read_command" {
  description = "AWS CLI command to read the config file"
  value       = "aws s3 cp s3://${local.bucket_name}/${var.config_file_key} -"
}

output "aws_cli_write_command" {
  description = "AWS CLI command template to write to config file"
  value       = "echo '{\"key\":\"value\"}' | aws s3 cp - s3://${local.bucket_name}/${var.config_file_key}"
}

output "curl_read_command" {
  description = "curl command to read the config file"
  value       = "curl https://${local.bucket_name}.storage.yandexcloud.net/${var.config_file_key}"
}