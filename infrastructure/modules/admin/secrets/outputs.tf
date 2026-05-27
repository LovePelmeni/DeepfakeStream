output "kms_key_id" {
  description = "ID of the created KMS symmetric key"
  value       = yandex_kms_symmetric_key.this.id
}

output "kms_key_name" {
  description = "Name of the created KMS symmetric key"
  value       = yandex_kms_symmetric_key.this.name
}

output "lockbox_secret_id" {
  description = "ID of the created Lockbox secret"
  value       = yandex_lockbox_secret.this.id
}

output "lockbox_secret_name" {
  description = "Name of the created Lockbox secret"
  value       = yandex_lockbox_secret.this.name
}

output "lockbox_initial_version_id" {
  description = "ID of the initial secret version (if created)"
  value       = try(yandex_lockbox_secret_version.initial[0].id, null)
}

output "has_initial_payload" {
  description = "Whether an initial payload was provided"
  value       = length(var.lockbox_version_payload) > 0
}

output "lockbox_secret_version_count" {
  description = "Number of secret versions created"
  value       = length(yandex_lockbox_secret_version.initial)
}