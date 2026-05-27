# S3 Bucket Outputs

output "s3_bucket_name" {
  description = "The name of the S3 bucket"
  value       = yandex_storage_bucket.s3_bucket.bucket
}

output "s3_bucket_id" {
  description = "The ID of the S3 bucket"
  value       = yandex_storage_bucket.s3_bucket.id
}

output "s3_bucket_domain_name" {
  description = "FQDN of the S3 bucket"
  value       = yandex_storage_bucket.s3_bucket.bucket_domain_name
}

output "s3_bucket_endpoint" {
  description = "The endpoint URL for the S3 bucket"
  value       = "https://${yandex_storage_bucket.s3_bucket.bucket_domain_name}"
}

# Aliases for what your root main.tf expects
output "bucket_name" {
  description = "Alias for s3_bucket_name"
  value       = yandex_storage_bucket.s3_bucket.bucket
}

output "bucket_id" {
  description = "Alias for s3_bucket_id"
  value       = yandex_storage_bucket.s3_bucket.id
}

output "bucket_domain_name" {
  description = "Alias for s3_bucket_domain_name"
  value       = yandex_storage_bucket.s3_bucket.bucket_domain_name
}