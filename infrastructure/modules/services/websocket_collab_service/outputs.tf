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

output "private_ip" {
  description = "Private IP address"
  value       = yandex_compute_instance.this.network_interface[0].ip_address
}

output "public_ip" {
  description = "Public IP address (if assign_public_ip = true)"
  value       = yandex_compute_instance.this.network_interface[0].nat_ip_address
}

# SSH connection information
output "ssh_user" {
  description = "SSH username"
  value       = var.ssh_user
}

output "ssh_public_key" {
  description = "SSH public key injected"
  value       = var.ssh_public_key
}

output "ssh_command_example" {
  description = "Example SSH command (replace <private-key-path> with your private key file)"
  value       = "ssh -i <private-key-path> ${var.ssh_user}@${yandex_compute_instance.this.network_interface[0].nat_ip_address != "" ? yandex_compute_instance.this.network_interface[0].nat_ip_address : yandex_compute_instance.this.network_interface[0].ip_address}"
}

# WebSocket endpoint (private)
output "websocket_endpoint" {
  description = "WebSocket endpoint URL (private IP)"
  value       = "ws://${yandex_compute_instance.this.network_interface[0].ip_address}:${var.app_port}"
}

# ALB resources (if created)
output "target_group_id" {
  description = "ID of the target group (if create_alb_resources = true)"
  value       = var.create_alb_resources ? yandex_alb_target_group.this[0].id : null
}

output "backend_group_id" {
  description = "ID of the backend group (if create_alb_resources = true)"
  value       = var.create_alb_resources ? yandex_alb_backend_group.this[0].id : null
}

output "has_alb_resources" {
  description = "Whether ALB resources were created"
  value       = var.create_alb_resources
}

output "zone" {
  description = "Availability zone of the instance"
  value       = var.zone
}