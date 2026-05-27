# ============================================
# BASIC INSTANCE INFORMATION
# ============================================

output "instance_ips" {
  description = "Private IP addresses of the compute instances"
  value       = yandex_compute_instance.this[*].network_interface[0].ip_address
}

output "instance_fqdns" {
  description = "FQDNs of the compute instances"
  value       = yandex_compute_instance.this[*].fqdn
}

output "instance_names" {
  description = "Names of the compute instances"
  value       = yandex_compute_instance.this[*].name
}

output "private_ips" {
  description = "Private IP addresses of Consul servers"
  value       = yandex_compute_instance.this[*].network_interface[0].ip_address
}

output "target_group_id" {
  description = "ID of the target group (if created)"
  value       = try(yandex_lb_target_group.this[0].id, null)
}

# ============================================
# SSH CONNECTION INFORMATION
# ============================================

output "ssh_user" {
  description = "SSH username for connecting to the instances"
  value       = var.ssh_user
}

output "ssh_public_key" {
  description = "SSH public key injected into the instances (use the matching private key)"
  value       = var.ssh_public_key
}

output "ssh_command_example" {
  description = "Example SSH command for the first instance (replace <private-key-path>)"
  value       = length(yandex_compute_instance.this) > 0 ? "ssh -i <private-key-path> ${var.ssh_user}@${yandex_compute_instance.this[0].network_interface[0].ip_address}" : "(no instances created)"
}

output "ssh_commands" {
  description = "SSH command templates for each instance (replace <private-key-path>)"
  value = {
    for idx, ip in yandex_compute_instance.this[*].network_interface[0].ip_address :
    "instance_${idx + 1}" => "ssh -i <private-key-path> ${var.ssh_user}@${ip}"
  }
}