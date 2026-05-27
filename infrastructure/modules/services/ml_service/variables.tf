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
  default     = "ml-server"
}

variable "zone" {
  description = "Availability zone (GPU zones: ru-central1-a, ru-central1-b, ru-central1-c)"
  type        = string
  default     = "ru-central1-a"
}

variable "platform_id" {
  description = "Platform ID – use 'gpu-standard-v3' for GPU instances"
  type        = string
  default     = "gpu-standard-v3"
}

variable "cores" {
  description = "Number of CPU cores"
  type        = number
  default     = 8
}

variable "core_fraction" {
  description = "Core fraction (5, 20, 50, 100)"
  type        = number
  default     = 100
}

variable "ram_gb" {
  description = "RAM in GB (minimum 32 GB recommended for ML workloads)"
  type        = number
  default     = 32
}

variable "gpus" {
  description = "Number of GPUs (0, 1, 2, 4)"
  type        = number
  default     = 1
}

variable "preemptible" {
  description = "Use preemptible instance (cheaper but may be terminated – not recommended for long training)"
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
  description = "Boot disk size in GB (recommended 100+ GB for ML tools and libraries)"
  type        = number
  default     = 100
}

variable "boot_disk_type" {
  description = "Boot disk type (network-ssd, network-hdd)"
  type        = string
  default     = "network-ssd"
}

# ----------------------------------------------------------------------
# DATA DISK (for datasets and models)
# ----------------------------------------------------------------------
variable "data_disk_enabled" {
  description = "Create and attach a secondary data disk"
  type        = bool
  default     = true
}

variable "data_disk_size_gb" {
  description = "Data disk size in GB (ignored if data_disk_enabled = false)"
  type        = number
  default     = 500
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
  description = "Custom image ID; if empty, uses the latest from os_image_family (use GPU‑enabled images for ML)"
  type        = string
  default     = ""
}

variable "os_image_family" {
  description = "OS image family (e.g., ubuntu-2204-lts, ubuntu-2204-lts-gpu)"
  type        = string
  default     = "ubuntu-2204-lts-gpu"
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
# USER DATA (optional cloud-init)
# ----------------------------------------------------------------------
variable "user_data" {
  description = "Cloud-init user data script (e.g., to install PyTorch, TensorFlow, or mount data disk)"
  type        = string
  default     = ""
}

# ----------------------------------------------------------------------
# NETWORK SECURITY
# ----------------------------------------------------------------------
variable "security_group_ids" {
  description = "List of security group IDs to attach (allow SSH, and any ML ports like Jupyter, TensorBoard)"
  type        = list(string)
  default     = []
}