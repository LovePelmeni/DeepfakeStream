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
  description = "Subnet ID for the instances"
  type        = string
}

# ----------------------------------------------------------------------
# INSTANCE CONFIGURATION
# ----------------------------------------------------------------------
variable "instance_name_prefix" {
  description = "Prefix for instance names (e.g., 'backend' -> 'backend-1', 'backend-2')"
  type        = string
  default     = "instance"
}

variable "instance_count" {
  description = "Number of instances to create"
  type        = number
  default     = 1
}

variable "zones" {
  description = "List of availability zones (round-robin distribution)"
  type        = list(string)
  default     = ["ru-central1-a"]
}

variable "cores" {
  description = "Number of CPU cores per instance"
  type        = number
  default     = 2
}

variable "core_fraction" {
  description = "Core fraction (5, 20, 50, 100)"
  type        = number
  default     = 100
}

variable "ram_bytes" {
  description = "RAM in bytes (e.g., 4 * 1024^3 = 4 GB)"
  type        = number
  default     = 4294967296  # 4 GB
}

variable "preemptible" {
  description = "Use preemptible instances (cheaper but may be terminated)"
  type        = bool
  default     = false
}

variable "assign_public_ip" {
  description = "Assign an ephemeral public IP to each instance"
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
  description = "Boot disk type (network-ssd, network-hdd, network-ssd-nonreplicated)"
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
# NETWORK SECURITY
# ----------------------------------------------------------------------
variable "security_group_ids" {
  description = "List of security group IDs to attach"
  type        = list(string)
  default     = []
}

# ----------------------------------------------------------------------
# TARGET GROUP (load balancer)
# ----------------------------------------------------------------------
variable "create_target_group" {
  description = "Create a target group containing all instances (for load balancer)"
  type        = bool
  default     = true
}

# ----------------------------------------------------------------------
# LABELS
# ----------------------------------------------------------------------
variable "purpose_label" {
  description = "Value for the 'purpose' label (e.g., 'backend', 'worker')"
  type        = string
  default     = "general"
}