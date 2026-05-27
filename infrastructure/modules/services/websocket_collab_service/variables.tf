# ----------------------------------------------------------------------
# REQUIRED
# ----------------------------------------------------------------------
variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for the instance"
  type        = string
}

# ----------------------------------------------------------------------
# INSTANCE CONFIGURATION
# ----------------------------------------------------------------------
variable "instance_name" {
  description = "Name of the instance"
  type        = string
  default     = "websocket-server"
}

variable "description" {
  description = "Description of the instance"
  type        = string
  default     = "WebSocket server for real-time collaboration"
}

variable "zone" {
  description = "Availability zone"
  type        = string
  default     = "ru-central1-a"
}

variable "platform_id" {
  description = "Platform ID (standard-v3, etc.)"
  type        = string
  default     = "standard-v3"
}

variable "cores" {
  description = "Number of CPU cores"
  type        = number
  default     = 2
}

variable "core_fraction" {
  description = "Core fraction (5, 20, 50, 100)"
  type        = number
  default     = 100
}

variable "ram_gb" {
  description = "RAM in GB"
  type        = number
  default     = 4
}

variable "preemptible" {
  description = "Use preemptible instance (cheaper but may be terminated)"
  type        = bool
  default     = false
}

variable "assign_public_ip" {
  description = "Assign an ephemeral public IP address"
  type        = bool
  default     = false
}

# ----------------------------------------------------------------------
# BOOT DISK
# ----------------------------------------------------------------------
variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 30
}

variable "boot_disk_type" {
  description = "Boot disk type (network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

# ----------------------------------------------------------------------
# OS IMAGE
# ----------------------------------------------------------------------
variable "image_id" {
  description = "Custom image ID; if empty, uses the latest from os_image_family"
  type        = string
  default     = ""
}

variable "os_image_family" {
  description = "OS image family (e.g., ubuntu-2204-lts)"
  type        = string
  default     = "ubuntu-2204-lts"
}

# ----------------------------------------------------------------------
# SSH ACCESS
# ----------------------------------------------------------------------
variable "ssh_user" {
  description = "SSH username"
  type        = string
  default     = "ubuntu"
}

variable "ssh_public_key" {
  description = "SSH public key to inject"
  type        = string
  sensitive   = true
}

# ----------------------------------------------------------------------
# USER DATA & METADATA
# ----------------------------------------------------------------------
variable "user_data" {
  description = "Cloud-init user data script (optional)"
  type        = string
  default     = ""
}

variable "extra_metadata" {
  description = "Additional metadata key-value pairs"
  type        = map(string)
  default     = {}
}

# ----------------------------------------------------------------------
# NETWORK SECURITY
# ----------------------------------------------------------------------
variable "security_group_ids" {
  description = "List of security group IDs to attach"
  type        = list(string)
  default     = []
}

# ----------------------------------------------------------------------
# APPLICATION CONFIGURATION
# ----------------------------------------------------------------------
variable "app_port" {
  description = "Port the WebSocket application listens on"
  type        = number
  default     = 8080
}

variable "health_check_path" {
  description = "HTTP path for health checks (used if create_alb_resources = true)"
  type        = string
  default     = "/health"
}

# ----------------------------------------------------------------------
# ALB INTEGRATION (optional)
# ----------------------------------------------------------------------
variable "create_alb_resources" {
  description = "Create ALB target group and backend group"
  type        = bool
  default     = false
}

# ----------------------------------------------------------------------
# LABELS
# ----------------------------------------------------------------------
variable "extra_labels" {
  description = "Additional labels for the instance, target group, and backend group"
  type        = map(string)
  default     = {}
}