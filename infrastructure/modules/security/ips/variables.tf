# Yandex Cloud provider credentials
variable "yc_token" {
  description = "Yandex Cloud OAuth token"
  type        = string
  sensitive   = true
}

variable "yc_cloud_id" {
  description = "Yandex Cloud cloud ID"
  type        = string
}

variable "yc_folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

# Region / zone
variable "yc_zone" {
  description = "Yandex Cloud availability zone"
  type        = string
  default     = "ru-central1-a"
}

# Image and network
variable "image_id" {
  description = "Image ID (Ubuntu 24.04 LTS recommended)"
  type        = string
  default     = "fd87va5cc00gaq2f5qfb"   # Ubuntu 24.04 in ru-central1-a
}

variable "subnet_id" {
  description = "Existing subnet ID where VMs will be placed"
  type        = string
}

# SSH access
variable "ssh_public_key" {
  description = "Path to SSH public key file (injected into VMs)"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "ssh_user" {
  description = "SSH username"
  type        = string
  default     = "ubuntu"
}

# VM instances definition (map)
variable "instances" {
  description = "Map of VM names to resources (cpu cores, ram GB, disk GB)"
  type = map(object({
    cpu  = number
    ram  = number
    disk = number
  }))
  default = {}   # Must be provided via .tfvars or CLI
}