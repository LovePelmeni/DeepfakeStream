# ============================================================
# YANDEX CLOUD AUTHENTICATION
# ============================================================
variable "yandex_cloud_id" {
  description = "Yandex Cloud ID"
  type        = string
  sensitive   = true
}

variable "yandex_cloud_folder_id" {
  description = "Yandex Cloud Folder ID"
  type        = string
  sensitive   = true
}

variable "yandex_cloud_zone" {
  description = "Yandex Cloud default zone"
  type        = string
  default     = "ru-central1-a"
}

variable "yandex_service_account_key_file" {
  description = "Path to Yandex Cloud service account key file"
  type        = string
  sensitive   = true
}

# ============================================================
# CONFIGURATION FILES
# ============================================================
variable "config_dir" {
  description = "Directory containing configuration JSON files"
  type        = string
  default     = "./configs"
}

# ============================================================
# S3 BACKEND (for Terraform state)
# ============================================================
variable "access_key" {
  description = "S3 access key for backend state storage"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "S3 secret key for backend state storage"
  type        = string
  sensitive   = true
}

variable "s3_bucket_name" {
  description = "S3 bucket name for Terraform state"
  type        = string
  default     = "vegamaps-infra-state-bucket"
}

# ============================================================
# SSL CERTIFICATE (for ALB)
# ============================================================
variable "ssl_certificate_id" {
  description = "SSL certificate ID for ALB"
  type        = string
  default     = null
}

# ============================================================
# ADMIN NETWORK (SSH access)
# ============================================================
variable "admin_ip_cidr" {
  description = "Allowed CIDR blocks for SSH access (admin networks)"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# ============================================================
# BASTION SSH KEYS
# ============================================================
variable "bastion_ssh_public_key" {
  description = "SSH public key for bastion host"
  type        = string
  sensitive   = true
}

# ============================================================
# OBSERVABILITY SSH KEYS
# ============================================================
variable "observability_ssh_public_key" {
  description = "SSH public key for observability server"
  type        = string
  sensitive   = true
}

# ============================================================
# SECURITY TOOLS (SIEM/SOAR) CONFIGURATION
# ============================================================
variable "security_tools_ssh_public_key" {
  description = "SSH public key for security tools VM"
  type        = string
  sensitive   = true
}

variable "security_tools_cores" {
  description = "CPU cores for SIEM/SOAR VM"
  type        = number
  default     = 8
}

variable "security_tools_ram_gb" {
  description = "RAM in GB for SIEM/SOAR VM"
  type        = number
  default     = 32
}

variable "security_tools_boot_disk_size" {
  description = "Boot disk size in GB for SIEM/SOAR VM"
  type        = number
  default     = 100
}

variable "security_tools_data_disk_enabled" {
  description = "Enable data disk for logs/events"
  type        = bool
  default     = true
}

variable "security_tools_data_disk_size" {
  description = "Data disk size in GB for SIEM/SOAR VM"
  type        = number
  default     = 500
}

# ============================================================
# GITHUB RUNNER CONFIGURATION
# ============================================================
variable "github_runner_ssh_public_key" {
  description = "SSH public key for GitHub runner VM"
  type        = string
  sensitive   = true
}

variable "github_runner_cores" {
  description = "CPU cores for GitHub runner"
  type        = number
  default     = 2
}

variable "github_runner_ram_gb" {
  description = "RAM in GB for GitHub runner"
  type        = number
  default     = 4
}

variable "github_runner_boot_disk_size" {
  description = "Boot disk size in GB for GitHub runner"
  type        = number
  default     = 40
}

variable "github_runner_data_disk_enabled" {
  description = "Enable data disk for GitHub runner (for Docker cache, etc.)"
  type        = bool
  default     = false
}

variable "github_runner_data_disk_size" {
  description = "Data disk size in GB for GitHub runner"
  type        = number
  default     = 100
}

variable "github_runner_user_data" {
  description = "Cloud-init script to install and register the GitHub runner"
  type        = string
  default     = ""
}

variable "github_runner_sg_name" {
  description = "Security group name for GitHub runner"
  type        = string
  default     = "prod-github-runner-sg"
}

# ============================================================
# BACKEND SERVICE SSH KEYS
# ============================================================
variable "backend_ssh_public_key" {
  description = "SSH public key for backend servers"
  type        = string
  sensitive   = true
}

# ============================================================
# FRONTEND SERVICE SSH KEYS
# ============================================================
variable "frontend_ssh_public_key" {
  description = "SSH public key for frontend servers"
  type        = string
  sensitive   = true
}

# ============================================================
# ML SERVICE SSH KEYS
# ============================================================
variable "ml_ssh_public_key" {
  description = "SSH public key for ML servers"
  type        = string
  sensitive   = true
}

# ============================================================
# WEBSOCKET SERVICE SSH KEYS
# ============================================================
variable "ws_ssh_public_key" {
  description = "SSH public key for WebSocket servers"
  type        = string
  sensitive   = true
}

# ============================================================
# SECURITY GROUP NAMES
# ============================================================
variable "alb_sg_name" {
  description = "ALB security group name"
  type        = string
  default     = "prod-alb-sg"
}

variable "observability_sg_name" {
  description = "Observability security group name"
  type        = string
  default     = "prod-observability-sg"
}

variable "backend_server_sg_name" {
  description = "Backend server security group name"
  type        = string
  default     = "prod-backend-sg"
}

variable "frontend_server_sg_name" {
  description = "Frontend server security group name"
  type        = string
  default     = "prod-frontend-sg"
}

variable "ml_service_sg_name" {
  description = "ML service security group name"
  type        = string
  default     = "prod-ml-sg"
}

variable "websocket_collab_sg_name" {
  description = "WebSocket collaboration security group name"
  type        = string
  default     = "prod-websocket-sg"
}

variable "database_sg_name" {
  description = "Database security group name"
  type        = string
  default     = "prod-database-sg"
}

variable "bastion_sg_name" {
  description = "Bastion security group name"
  type        = string
  default     = "prod-bastion-sg"
}

variable "consul_server_sg_name" {
  description = "Consul server security group name"
  type        = string
  default     = "prod-consul-server-sg"
}

variable "security_tools_sg_name" {
  description = "Security tools security group name"
  type        = string
  default     = "prod-security-tools-sg"
}

# ============================================================
# CONTAINER REGISTRY
# ============================================================
variable "registry_name" {
  description = "Container registry name"
  type        = string
  default     = "vegamaps-prod-registry"
}

variable "registry_labels" {
  description = "Container registry labels"
  type        = map(string)
  default = {
    environment = "prod"
    managed_by  = "terraform"
  }
}

variable "create_registry_admin" {
  description = "Create registry admin service account"
  type        = bool
  default     = false
}

# ============================================================
# CONSUL SERVICE MESH CONFIGURATION
# ============================================================
variable "consul_servers_count" {
  description = "Number of Consul server nodes (must be odd)"
  type        = number
  default     = 3
  validation {
    condition     = var.consul_servers_count % 2 == 1
    error_message = "Consul server count must be odd"
  }
}

variable "consul_server_cores" {
  description = "CPU cores per Consul server"
  type        = number
  default     = 2
}

variable "consul_server_memory_gb" {
  description = "Memory per Consul server (GB)"
  type        = number
  default     = 4
}

variable "consul_server_disk_size" {
  description = "Boot disk size for Consul server (GB)"
  type        = number
  default     = 50
}

variable "consul_server_disk_type" {
  description = "Disk type for Consul server"
  type        = string
  default     = "network-ssd"
}

variable "consul_server_platform_id" {
  description = "Platform ID for Consul server"
  type        = string
  default     = "standard-v3"
}

variable "consul_create_load_balancer" {
  description = "Create an internal load balancer for Consul"
  type        = bool
  default     = true
}


# ============================================================
# NOMAD SERVER CLUSTER CONFIGURATION
# ============================================================
variable "nomad_servers_count" {
  description = "Number of Nomad server nodes (must be odd for HA)"
  type        = number
  default     = 3
  validation {
    condition     = var.nomad_servers_count % 2 == 1
    error_message = "Nomad server count must be odd"
  }
}

variable "nomad_server_cores" {
  description = "CPU cores per Nomad server"
  type        = number
  default     = 2
}

variable "nomad_server_memory_gb" {
  description = "Memory per Nomad server (GB)"
  type        = number
  default     = 4
}

variable "nomad_server_disk_size" {
  description = "Boot disk size for Nomad server (GB)"
  type        = number
  default     = 50
}

variable "nomad_server_disk_type" {
  description = "Disk type for Nomad server"
  type        = string
  default     = "network-ssd"
}

variable "nomad_server_sg_name" {
  description = "Security group name for Nomad servers"
  type        = string
  default     = "prod-nomad-server-sg"
}