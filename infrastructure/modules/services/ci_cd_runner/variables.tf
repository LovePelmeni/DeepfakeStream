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
variable "instance_name_prefix" {
  description = "Prefix for the instance name (e.g., 'github-runner')"
  type        = string
  default     = "github-runner"
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
  description = "Number of CPU cores (GitHub runners typically need 2-4 cores)"
  type        = number
  default     = 2
}

variable "core_fraction" {
  description = "Core fraction (5, 20, 50, 100)"
  type        = number
  default     = 100
}

variable "ram_gb" {
  description = "RAM in GB (recommended at least 4 GB for runners)"
  type        = number
  default     = 4
}

variable "preemptible" {
  description = "Use preemptible instance (cheaper but may be terminated – not ideal for long jobs)"
  type        = bool
  default     = false
}

variable "assign_public_ip" {
  description = "Assign an ephemeral public IP (required for GitHub runner to communicate outbound)"
  type        = bool
  default     = true
}

variable "static_ip_address" {
  description = "Static public IP address (if provided, assign_public_ip is ignored)"
  type        = string
  default     = null
}

# ----------------------------------------------------------------------
# BOOT DISK
# ----------------------------------------------------------------------
variable "boot_disk_size_gb" {
  description = "Boot disk size in GB (runner tools and temporary files)"
  type        = number
  default     = 40
}

variable "boot_disk_type" {
  description = "Boot disk type (network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

# ----------------------------------------------------------------------
# DATA DISK (optional – for persistent cache or Docker storage)
# ----------------------------------------------------------------------
variable "data_disk_enabled" {
  description = "Create and attach a secondary data disk"
  type        = bool
  default     = false
}

variable "data_disk_size_gb" {
  description = "Data disk size in GB (ignored if data_disk_enabled = false)"
  type        = number
  default     = 100
}

variable "data_disk_type" {
  description = "Data disk type (network-ssd, network-hdd)"
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
  description = "SSH public key to inject (for debugging or manual setup)"
  type        = string
  sensitive   = true
}

# ----------------------------------------------------------------------
# USER DATA (optional cloud-init – e.g., to install and register runner)
# ----------------------------------------------------------------------
variable "user_data" {
  description = "Cloud-init user data script (e.g., install GitHub runner, register with token)"
  type        = string
  default     = ""
}

# ----------------------------------------------------------------------
# NETWORK SECURITY
# ----------------------------------------------------------------------
variable "security_group_ids" {
  description = "List of security group IDs to attach (must allow outbound HTTPS to GitHub API)"
  type        = list(string)
  default     = []
}