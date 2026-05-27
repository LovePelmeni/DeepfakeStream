output "instance_id" {
  description = "ID of the compute instance"
  value       = yandex_compute_instance.this.id
}

output "instance_name" {
  description = "Name of the compute instance"
  value       = yandex_compute_instance.this.name
}

output "instance_fqdn" {
  description = "FQDN of the instance"
  value       = yandex_compute_instance.this.fqdn
}

output "external_ip" {
  description = "Public IP address (if assign_public_ip = true)"
  value       = yandex_compute_instance.this.network_interface[0].nat_ip_address
}

output "internal_ip" {
  description = "Private IP address"
  value       = yandex_compute_instance.this.network_interface[0].ip_address
}

# SSH connection information
output "ssh_user" {
  description = "SSH username"
  value       = var.ssh_user
}

output "ssh_public_key" {
  description = "SSH public key injected into the instance"
  value       = var.ssh_public_key
}

output "ssh_command_example" {
  description = "Example SSH command (replace <private-key-path> with your private key file)"
  value       = "ssh -i <private-key-path> ${var.ssh_user}@${yandex_compute_instance.this.network_interface[0].nat_ip_address != "" ? yandex_compute_instance.this.network_interface[0].nat_ip_address : yandex_compute_instance.this.network_interface[0].ip_address}"
}

# Data disk information (if created)
output "data_disk_id" {
  description = "ID of the data disk (if enabled)"
  value       = var.data_disk_enabled ? yandex_compute_disk.data_disk[0].id : null
}