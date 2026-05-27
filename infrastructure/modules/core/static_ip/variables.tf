variable "name" {
  description = "Name of the static IP address"
  type        = string
  
  validation {
    condition     = can(regex("^[a-z][a-z0-9-]*$", var.name))
    error_message = "Name must start with a letter and contain only lowercase letters, numbers, and hyphens."
  }
}

variable "folder_id" {
  description = "Yandex Cloud folder ID where the IP will be created"
  type        = string
}

variable "zone_id" {
  description = "Availability zone for the IP address"
  type        = string
  default     = "ru-central1-a"
  
  validation {
    condition     = contains(["ru-central1-a", "ru-central1-b", "ru-central1-d"], var.zone_id)
    error_message = "Zone must be one of: ru-central1-a, ru-central1-b, ru-central1-d."
  }
}

variable "description" {
  description = "Description of the static IP address"
  type        = string
  default     = ""
}

variable "labels" {
  description = "Labels to assign to the static IP"
  type        = map(string)
  default     = {}
}

variable "deletion_protection" {
  description = "Enable deletion protection for the static IP"
  type        = bool
  default     = true
}

# DNS Configuration
variable "dns_config" {
  description = "DNS configuration for the static IP"
  type = object({
    zone_id       = string          # Yandex Cloud DNS zone ID
    subdomain     = string          # Subdomain (e.g., "bastion", "api")
    domain        = string          # Domain name (e.g., "vegamaps.com")
    ttl           = number          # TTL in seconds
    description   = optional(string, "DNS record for static IP")
    create_ptr    = optional(bool, false)  # Create PTR record for reverse DNS
    ptr_ttl       = optional(number, 300)  # TTL for PTR record
  })
  default = null
}

variable "create_dns_record" {
  description = "Whether to create DNS record for this IP"
  type        = bool
  default     = false
}

variable "dns_zone_folder_id" {
  description = "Folder ID where DNS zone exists (if different from IP folder)"
  type        = string
  default     = null
}

variable "environment" {
  description = "Environment name (dev, prod, etc.) for DNS naming"
  type        = string
  default     = null
}

variable "service_name" {
  description = "Service name (bastion, api, frontend, etc.)"
  type        = string
  default     = null
}