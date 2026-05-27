output "quarantine_bucket_name" {
  description = "Name of the quarantine bucket"
  value       = yandex_storage_bucket.quarantine.bucket
}

output "scanner_function_id" {
  description = "ID of the antivirus scanner function"
  value       = yandex_function.scanner.id
}

output "scanner_function_name" {
  description = "Name of the antivirus scanner function"
  value       = yandex_function.scanner.name
}

output "trigger_count" {
  description = "Number of upload triggers created"
  value       = length(yandex_function_trigger.upload_trigger)
}

output "trigger_ids" {
  description = "IDs of all upload triggers"
  value       = yandex_function_trigger.upload_trigger[*].id
}