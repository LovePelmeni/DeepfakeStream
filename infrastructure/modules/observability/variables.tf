# ----------------------------------------------------------------------
# REQUIRED
# ----------------------------------------------------------------------
variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "environment" {
  description = "Environment name (e.g., dev, staging, prod)"
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
  default     = "compute-instance"
}

variable "zone" {
  description = "Availability zone"
  type        = string
  default     = "ru-central1-a"
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

variable "ram_memory_bytes" {
  description = "RAM in bytes (alternative to ram_gb)"
  type        = number
  default     = null
}

variable "preemptible" {
  description = "Use preemptible instance (cheaper but may be terminated)"
  type        = bool
  default     = false
}

variable "assign_public_ip" {
  description = "Assign a public IPv4 address"
  type        = bool
  default     = true
}

variable "static_ip_address" {
  description = "Static internal IP address (optional, must be within subnet range)"
  type        = string
  default     = null
}

# ----------------------------------------------------------------------
# BOOT DISK
# ----------------------------------------------------------------------
variable "boot_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "boot_disk_type" {
  description = "Boot disk type (network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

# ----------------------------------------------------------------------
# DATA DISK (optional)
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
# NETWORK SECURITY
# ----------------------------------------------------------------------
variable "security_group_ids" {
  description = "List of security group IDs to attach"
  type        = list(string)
  default     = []
}