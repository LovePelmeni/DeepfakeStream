# ----------------------------------------------------------------------
# IDs of all instances (map)
# ----------------------------------------------------------------------
output "instance_ids" {
  description = "Map of instance names to instance IDs"
  value = {
    for name, inst in yandex_compute_instance.soc_vm :
    name => inst.id
  }
}

# ----------------------------------------------------------------------
# Names of all instances (list)
# ----------------------------------------------------------------------
output "instance_names" {
  description = "List of all instance names"
  value = keys(yandex_compute_instance.soc_vm)
}

# ----------------------------------------------------------------------
# Private IP addresses (map)
# ----------------------------------------------------------------------
output "private_ips" {
  description = "Map of instance names to private IP addresses"
  value = {
    for name, inst in yandex_compute_instance.soc_vm :
    name => inst.network_interface[0].ip_address
  }
}

# ----------------------------------------------------------------------
# Public (static) IP addresses (map)
# ----------------------------------------------------------------------
output "public_ips" {
  description = "Map of instance names to static public IP addresses"
  value = {
    for name, addr in yandex_vpc_address.static_ip :
    name => addr.external_ipv4_address[0].address
  }
}

# ----------------------------------------------------------------------
# First instance helpers (alphabetical order)
# ----------------------------------------------------------------------
output "first_private_ip" {
  description = "Private IP of the first instance (sorted by name)"
  value = length(yandex_compute_instance.soc_vm) > 0 ? yandex_compute_instance.soc_vm[sort(keys(yandex_compute_instance.soc_vm))[0]].network_interface[0].ip_address : null
}

output "first_public_ip" {
  description = "Public IP of the first instance (sorted by name)"
  value = length(yandex_vpc_address.static_ip) > 0 ? yandex_vpc_address.static_ip[sort(keys(yandex_vpc_address.static_ip))[0]].external_ipv4_address[0].address : null
}

# ----------------------------------------------------------------------
# SSH info (from variables)
# ----------------------------------------------------------------------
output "ssh_user" {
  description = "SSH username"
  value       = var.ssh_user
}

output "ssh_public_key_path" {
  description = "SSH public key file path"
  value       = var.ssh_public_key
}

output "ssh_command_example" {
  description = "Example SSH command for the first instance"
  value = length(yandex_compute_instance.soc_vm) > 0 ? "ssh -i <private-key-path> ${var.ssh_user}@${output.first_public_ip.value}" : "(no instances)"
}

# ----------------------------------------------------------------------
# Detailed map of instances (for automation)
# ----------------------------------------------------------------------
output "instances_map" {
  description = "Detailed map of instance names to private IP, public IP, zone, ID, CPU, RAM, disk"
  value = {
    for name, inst in yandex_compute_instance.soc_vm :
    name => {
      private_ip   = inst.network_interface[0].ip_address
      public_ip    = yandex_vpc_address.static_ip[name].external_ipv4_address[0].address
      instance_id  = inst.id
      zone         = inst.zone
      cpu          = var.instances[name].cpu
      ram_gb       = var.instances[name].ram
      disk_gb      = var.instances[name].disk
    }
  }
}