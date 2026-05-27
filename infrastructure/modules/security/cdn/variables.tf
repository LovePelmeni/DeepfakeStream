variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for CDN resources"
  type        = string
  default     = "cdn"
}

# CDN Resource basic settings
variable "cname" {
  description = "CNAME for CDN resource"
  type        = string
}

variable "active" {
  description = "Whether CDN resource is active"
  type        = bool
  default     = true
}

variable "secondary_hostnames" {
  description = "List of secondary hostnames"
  type        = list(string)
  default     = []
}

# Origin settings
variable "origin_source" {
  description = "Origin source (bucket name or IP address)"
  type        = string
}

variable "origin_protocol" {
  description = "Origin protocol: 'http', 'https', or 'match'"
  type        = string
  default     = "https"
  
  validation {
    condition     = contains(["http", "https", "match"], var.origin_protocol)
    error_message = "Origin protocol must be http, https, or match"
  }
}

# SSL Certificate
variable "ssl_certificate_type" {
  description = "SSL certificate type: 'not_used', 'cm', or 'imported'"
  type        = string
  default     = "not_used"
  
  validation {
    condition     = contains(["not_used", "cm", "imported"], var.ssl_certificate_type)
    error_message = "Certificate type must be not_used, cm, or imported"
  }
}

variable "ssl_certificate_id" {
  description = "SSL certificate ID from Certificate Manager (required if type is 'cm')"
  type        = string
  default     = null
}

# DNS settings
variable "create_dns_record" {
  description = "Whether to create DNS CNAME record"
  type        = bool
  default     = false
}

variable "dns_zone_id" {
  description = "DNS zone ID for custom domain (required if create_dns_record is true)"
  type        = string
  default     = null
}

variable "dns_record_name" {
  description = "DNS record name (defaults to cname if not specified)"
  type        = string
  default     = null
}

variable "dns_ttl" {
  description = "DNS record TTL in seconds"
  type        = number
  default     = 300
}

# Labels
variable "labels" {
  description = "Labels for CDN resources"
  type        = map(string)
  default     = {}
}