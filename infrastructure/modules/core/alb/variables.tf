variable "name_prefix" {
  description = "Prefix for ALB resources"
  type        = string
}

variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "network_id" {
  description = "VPC network ID"
  type        = string
}

variable "subnet_ids" {
  description = "List of subnet IDs for ALB allocation"
  type        = list(string)
}

variable "security_group_ids" {
  description = "List of security group IDs for the ALB"
  type        = list(string)
  default     = []
}

variable "ssl_certificate_id" {
  description = "SSL certificate ID for HTTPS listener (not currently used)"
  type        = string
  default     = null
}

variable "labels" {
  description = "Labels for ALB resources"
  type        = map(string)
  default     = {}
}

variable "tiers" {
  description = "Tier configuration for routing"
  type = map(object({
    backend_group_id = string
  }))
  default = {}
}

variable "default_backend_group_id" {
  description = "Default backend group ID for unmatched routes"
  type        = string
  default     = null
}