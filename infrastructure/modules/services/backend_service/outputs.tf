output "instance_ids" {
  description = "IDs of all instances"
  value       = yandex_compute_instance.this[*].id
}

output "instance_names" {
  description = "Names of all instances"
  value       = yandex_compute_instance.this[*].name
}

output "private_ips" {
  description = "Private IP addresses of all instances"
  value       = yandex_compute_instance.this[*].network_interface[0].ip_address
}

output "public_ips" {
  description = "Public IP addresses (if assign_public_ip = true)"
  value       = yandex_compute_instance.this[*].network_interface[0].nat_ip_address
}

output "first_private_ip" {
  description = "Private IP of the first instance (useful for single-instance cases)"
  value       = var.instance_count > 0 ? yandex_compute_instance.this[0].network_interface[0].ip_address : null
}

output "target_group_id" {
  description = "ID of the target group (if create_target_group = true)"
  value       = var.create_target_group && var.instance_count > 0 ? yandex_lb_target_group.this[0].id : null
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
  description = "Example SSH command for the first instance (replace <private-key-path>)"
  value       = var.instance_count > 0 ? "ssh -i <private-key-path> ${var.ssh_user}@${yandex_compute_instance.this[0].network_interface[0].nat_ip_address != "" ? yandex_compute_instance.this[0].network_interface[0].nat_ip_address : yandex_compute_instance.this[0].network_interface[0].ip_address}" : "(no instances)"
}

# Map of instances for automation
output "instances_map" {
  description = "Map of instance names to their private IP and zone"
  value = {
    for idx, inst in yandex_compute_instance.this :
    inst.name => {
      private_ip = inst.network_interface[0].ip_address
      instance_id = inst.id
      zone = inst.zone
    }
  }
}