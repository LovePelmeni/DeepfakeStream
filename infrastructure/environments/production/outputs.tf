# ============================================================
# OUTPUTS
# ============================================================
output "alb_ip" {
  description = "ALB external IP address"
  value       = module.alb_static_ip.ip_address
}

output "bastion_ip" {
  description = "Bastion external IP address"
  value       = module.bastion_static_ip.ip_address
}

output "observability_ip" {
  description = "Observability server external IP address"
  value       = module.observability_static_ip.ip_address
}

output "security_tools_private_ip" {
  description = "Private IP of the security tools (SIEM/SOAR) VM"
  value       = module.security_tools.private_ip
}

output "security_tools_ssh_command" {
  description = "SSH command to access security tools VM (via bastion)"
  value       = "ssh -J ${module.bastion.public_ip} ${module.security_tools.ssh_user}@${module.security_tools.private_ip}"
}

output "github_runner_public_ip" {
  description = "Public IP of the GitHub runner"
  value       = module.github_runner.public_ip
}

output "github_runner_private_ip" {
  description = "Private IP of the GitHub runner"
  value       = module.github_runner.private_ip
}

output "github_runner_ssh_command" {
  description = "SSH command for GitHub runner"
  value       = "ssh -i <private-key> ${module.github_runner.ssh_user}@${module.github_runner.public_ip}"
}

output "database_hosts" {
  description = "PostgreSQL cluster hosts"
  value       = yandex_mdb_postgresql_cluster.shared.host[*].fqdn
}

output "database_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "registry_id" {
  description = "Container registry ID"
  value       = yandex_container_registry.prod_registry.id
}

output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = module.s3_bucket.bucket_name
}

# Convenience outputs for free/paid tier VMs
output "backend_free_ips" {
  description = "Private IPs of free tier backend instances"
  value       = module.backend_free.private_ips
}

output "frontend_free_ip" {
  description = "Private IP of free tier frontend instance"
  value       = module.frontend_free.private_ip
}

output "ml_free_ip" {
  description = "Private IP of free tier ML instance"
  value       = module.ml_free.private_ip
}

output "websocket_free_ip" {
  description = "Private IP of free tier WebSocket instance"
  value       = module.websocket_free.private_ip
}

output "backend_paid_ips" {
  description = "Private IPs of paid tier backend instances"
  value       = module.backend_paid.private_ips
}

output "frontend_paid_ip" {
  description = "Private IP of paid tier frontend instance"
  value       = module.frontend_paid.private_ip
}

output "ml_paid_ip" {
  description = "Private IP of paid tier ML instance"
  value       = module.ml_paid.private_ip
}

output "websocket_paid_ip" {
  description = "Private IP of paid tier WebSocket instance"
  value       = module.websocket_paid.private_ip
}

output "consul_server_ips" {
  description = "Private IPs of Consul servers"
  value       = module.consul.private_ips
}

# ============================================================
# NOMAD CLUSTER OUTPUTS
# ============================================================
output "nomad_free_private_ips" {
  description = "Private IPs of free tier Nomad servers"
  value       = module.nomad_free.private_ips
}

output "nomad_paid_private_ips" {
  description = "Private IPs of paid tier Nomad servers"
  value       = module.nomad_paid.private_ips
}

# ============================================================
# NOMAD FREE TIER CLUSTER
# ============================================================
variable "nomad_free_servers_count" {
  description = "Number of Nomad servers in free tier (must be odd)"
  type        = number
  default     = 1
  validation {
    condition     = var.nomad_free_servers_count % 2 == 1
    error_message = "Nomad server count must be odd."
  }
}

variable "nomad_free_server_cores" {
  description = "CPU cores per Nomad server (free tier)"
  type        = number
  default     = 1
}

variable "nomad_free_server_memory_gb" {
  description = "Memory (GB) per Nomad server (free tier)"
  type        = number
  default     = 2
}

variable "nomad_free_server_disk_size" {
  description = "Boot disk size (GB) for free tier Nomad server"
  type        = number
  default     = 30
}

variable "nomad_free_server_disk_type" {
  description = "Disk type for free tier Nomad server"
  type        = string
  default     = "network-hdd"
}

variable "nomad_free_preemptible" {
  description = "Use preemptible instances for free tier Nomad"
  type        = bool
  default     = true
}

# ============================================================
# NOMAD PAID TIER CLUSTER
# ============================================================
variable "nomad_paid_servers_count" {
  description = "Number of Nomad servers in paid tier (must be odd)"
  type        = number
  default     = 3
  validation {
    condition     = var.nomad_paid_servers_count % 2 == 1
    error_message = "Nomad server count must be odd."
  }
}

variable "nomad_paid_server_cores" {
  description = "CPU cores per Nomad server (paid tier)"
  type        = number
  default     = 2
}

variable "nomad_paid_server_memory_gb" {
  description = "Memory (GB) per Nomad server (paid tier)"
  type        = number
  default     = 4
}

variable "nomad_paid_server_disk_size" {
  description = "Boot disk size (GB) for paid tier Nomad server"
  type        = number
  default     = 50
}

variable "nomad_paid_server_disk_type" {
  description = "Disk type for paid tier Nomad server"
  type        = string
  default     = "network-ssd"
}

variable "nomad_paid_preemptible" {
  description = "Use preemptible instances for paid tier Nomad (not recommended)"
  type        = bool
  default     = false
}