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
# CONFIGURATION FILES (JSON based configs)
# ============================================================
variable "config_dir" {
  description = "Directory containing configuration files"
  type        = string
  default     = "./config"
}

variable "free_tier_config_file" {
  description = "Path to free tier configuration file (JSON)"
  type        = string
  default     = null
}

variable "paid_tier_config_file" {
  description = "Path to paid tier configuration file (JSON)"
  type        = string
  default     = null
}

variable "global_config_file" {
  description = "Path to global configuration file (JSON)"
  type        = string
  default     = null
}

# ============================================================
# LOAD CONFIGURATIONS
# ============================================================
locals {
  # Load global config with fallback
  global_config_raw = try(
    var.global_config_file != null ? file(var.global_config_file) : file("${var.config_dir}/defaults/global.json"),
    file("${path.module}/config/defaults/global.json"),
    jsonencode({
      environment = "prod"
      defaults = {
        core_fraction = 100
        boot_disk_type = "network-ssd"
        assign_public_ip = false
        preemptible = false
      }
      observability = {
        cores = 8
        memory_gb = 32
        boot_disk_size = 100
        data_disk_size = 500
        data_disk_enabled = true
        services = {
          grafana = true
          prometheus = true
          loki = true
          tempo = true
          mlflow = true
          airflow = true
        }
      }
      postgresql = {
        preset_id = "s2.micro"
        disk_size = 100
        max_connections = 200
        version = 15
      }
      bastion = {
        cores = 2
        memory_gb = 4
        disk_size = 20
      }
      waf = {
        paranoia_level = 2
        anomaly_threshold = 25
        mode = "BLOCK"
      }
    })
  )
  global_config = jsondecode(local.global_config_raw)
  
  # Load free tier config
  free_tier_raw = try(
    var.free_tier_config_file != null ? file(var.free_tier_config_file) : file("${var.config_dir}/free/tier.json"),
    file("${path.module}/config/free/tier.json"),
    jsonencode({
      zone = "ru-central1-a"
      subnet_key = "ru-central1-a"
      backend = {
        cores = 2
        memory_gb = 4
        disk_size = 30
        instance_count = 1
      }
      frontend = {
        cores = 2
        memory_gb = 4
        disk_size = 30
      }
      ml = {
        cores = 4
        memory_gb = 8
        disk_size = 50
        gpus = 0
        platform = "standard-v3"
      }
      websocket = {
        cores = 2
        memory_gb = 4
        disk_size = 30
      }
    })
  )
  free_tier = jsondecode(local.free_tier_raw)
  
  # Load paid tier config
  paid_tier_raw = try(
    var.paid_tier_config_file != null ? file(var.paid_tier_config_file) : file("${var.config_dir}/paid/tier.json"),
    file("${path.module}/config/paid/tier.json"),
    jsonencode({
      zone = "ru-central1-b"
      subnet_key = "ru-central1-b"
      backend = {
        cores = 8
        memory_gb = 16
        disk_size = 50
        instance_count = 3
      }
      frontend = {
        cores = 4
        memory_gb = 8
        disk_size = 50
      }
      ml = {
        cores = 8
        memory_gb = 32
        disk_size = 100
        gpus = 1
        platform = "gpu-standard-v3"
      }
      websocket = {
        cores = 4
        memory_gb = 8
        disk_size = 50
      }
    })
  )
  paid_tier = jsondecode(local.paid_tier_raw)
  
  environment = local.global_config.environment
}

# ============================================================
# S3 BACKEND
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
# SSL CERTIFICATE
# ============================================================
variable "ssl_certificate_id" {
  description = "SSL certificate ID for ALB"
  type        = string
  default     = null
}

# ============================================================
# ORGANIZATION
# ============================================================
variable "yandex_organization_id" {
  description = "Yandex Cloud organization ID"
  type        = string
  default     = ""
}

# ============================================================
# BASTION SSH KEYS
# ============================================================
variable "bastion_ssh_public_key" {
  description = "SSH public key for bastion host"
  type        = string
  sensitive   = true
}

variable "bastion_admin_ip_cidr" {
  description = "Admin IP CIDR for bastion SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "bastion_metadata_ip_cidr" {
  description = "Metadata IP CIDR for bastion"
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "bastion_os_image" {
  description = "OS image for bastion host"
  type        = string
  default     = "ubuntu-2204-lts"
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
# BACKEND SERVICE SSH KEYS
# ============================================================
variable "backend_ssh_public_key" {
  description = "SSH public key for backend servers"
  type        = string
  sensitive   = true
}

variable "backend_os_image" {
  description = "OS image for backend servers"
  type        = string
  default     = "ubuntu-2204-lts"
}

# ============================================================
# FRONTEND SERVICE SSH KEYS
# ============================================================
variable "frontend_ssh_public_key" {
  description = "SSH public key for frontend servers"
  type        = string
  sensitive   = true
}

variable "frontend_os_image" {
  description = "OS image for frontend servers"
  type        = string
  default     = "ubuntu-2204-lts"
}

# ============================================================
# ML SERVICE SSH KEYS
# ============================================================
variable "ml_ssh_public_key" {
  description = "SSH public key for ML servers"
  type        = string
  sensitive   = true
}

variable "ml_os_image" {
  description = "OS image for ML servers"
  type        = string
  default     = "ubuntu-2204-lts"
}

# ============================================================
# WEBSOCKET SERVICE SSH KEYS
# ============================================================
variable "ws_ssh_public_key" {
  description = "SSH public key for WebSocket servers"
  type        = string
  sensitive   = true
}

variable "ws_os_image" {
  description = "OS image for WebSocket servers"
  type        = string
  default     = "ubuntu-2204-lts"
}

# ============================================================
# SECURITY GROUPS NAMES
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

# ============================================================
# SIEM/SOAR CONFIGURATION
# ============================================================
variable "audit_log_retention_days" {
  description = "Audit log retention in days"
  type        = number
  default     = 365
}

variable "vpc_flow_log_retention_days" {
  description = "VPC flow log retention in days"
  type        = number
  default     = 90
}

variable "vm_secure_log_retention_days" {
  description = "VM secure log retention in days"
  type        = number
  default     = 180
}

variable "soc_bot_token" {
  description = "Telegram bot token for SOC alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "soc_chat_id" {
  description = "Telegram chat ID for SOC alerts"
  type        = string
  sensitive   = true
  default     = ""
}

variable "soc_rule_sg_on" {
  description = "Enable security group rule detection"
  type        = bool
  default     = false
}

variable "soc_rule_bucket_on" {
  description = "Enable bucket public access detection"
  type        = bool
  default     = false
}

variable "soc_rule_secret_on" {
  description = "Enable secret permission detection"
  type        = bool
  default     = false
}

variable "soc_del_rule_on" {
  description = "Enable automatic security group rule deletion"
  type        = bool
  default     = false
}

variable "soc_del_perm_secret_on" {
  description = "Enable automatic secret permission revocation"
  type        = bool
  default     = false
}

# ============================================================
# OUTPUT CONFIGURATION
# ============================================================
output "loaded_configs" {
  description = "Loaded configuration from JSON files"
  value = {
    environment = local.environment
    free_tier   = local.free_tier
    paid_tier   = local.paid_tier
    global      = local.global_config
  }
  sensitive = false
}