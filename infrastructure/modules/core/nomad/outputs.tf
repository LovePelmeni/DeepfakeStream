output "instance_ids" {
  description = "IDs of all Nomad instances"
  value       = yandex_compute_instance.this[*].id
}

output "private_ips" {
  description = "Private IP addresses of all Nomad instances"
  value       = yandex_compute_instance.this[*].network_interface[0].ip_address
}

output "public_ips" {
  description = "Public IP addresses (if any) of Nomad instances"
  value       = yandex_compute_instance.this[*].network_interface[0].nat_ip_address
}

output "instance_names" {
  description = "Names of all Nomad instances"
  value       = yandex_compute_instance.this[*].name
}

output "ssh_user" {
  description = "SSH username"
  value       = var.ssh_user
}

output "ssh_public_key" {
  description = "SSH public key injected"
  value       = var.ssh_public_key
}