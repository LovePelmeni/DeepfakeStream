# ----------------------------------------------------------------------
# REQUIRED
# ----------------------------------------------------------------------
variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "environment" {
  description = "Environment name (free, paid, prod, etc.)"
  type        = string
}

variable "private_subnet_ids" {
  description = "Map of subnet IDs per zone"
  type        = map(string)
}

variable "security_group_id" {
  description = "Security group ID to attach"
  type        = string
}

# ----------------------------------------------------------------------
# NETWORK & ZONES
# ----------------------------------------------------------------------
variable "zones" {
  description = "List of availability zones (only used for subnet key if needed)"
  type        = list(string)
  default     = ["ru-central1-a", "ru-central1-b", "ru-central1-c"]
}

variable "domain" {
  description = "DNS domain for hostnames"
  type        = string
  default     = "nomad.internal"
}

# ----------------------------------------------------------------------
# INSTANCE RESOURCES (prefixed names)
# ----------------------------------------------------------------------
variable "instance_count" {
  description = "Number of Nomad server instances"
  type        = number
  default     = 3
}

variable "instance_cores" {
  description = "vCPU cores per instance"
  type        = number
  default     = 2
}

variable "instance_ram_gb" {
  description = "RAM in GB per instance"
  type        = number
  default     = 4
}

variable "instance_core_fraction" {
  description = "CPU core fraction (5,20,50,100)"
  type        = number
  default     = 100
}

variable "instance_disk_size_gb" {
  description = "Boot disk size in GB"
  type        = number
  default     = 50
}

variable "instance_disk_type" {
  description = "Disk type (network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

variable "instance_platform_id" {
  description = "Platform ID (standard-v3, etc.)"
  type        = string
  default     = "standard-v3"
}

variable "preemptible" {
  description = "Use preemptible instances"
  type        = bool
  default     = false
}

# ----------------------------------------------------------------------
# OS IMAGE
# ----------------------------------------------------------------------
variable "image_id" {
  description = "Custom image ID; empty uses latest from os_family"
  type        = string
  default     = ""
}

variable "os_family" {
  description = "OS family for image lookup (e.g., ubuntu-2204-lts)"
  type        = string
  default     = "ubuntu-2204-lts"
}

variable "image_folder_id" {
  description = "Folder where the image resides"
  type        = string
  default     = "standard-images"
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
  description = "SSH public key"
  type        = string
  sensitive   = true
}

# ----------------------------------------------------------------------
# TARGET GROUP (optional – not used by default)
# ----------------------------------------------------------------------
variable "create_target_group" {
  description = "Create a target group for load balancing"
  type        = bool
  default     = false
}