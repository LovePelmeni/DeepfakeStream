# Basic Configuration
variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for WAF resources"
  type        = string
  default     = "waf"
}

variable "description" {
  description = "Description of the WAF profile"
  type        = string
  default     = "WAF profile for web application protection"
}

variable "enabled" {
  description = "Enable the WAF profile"
  type        = bool
  default     = true
}

# OWASP Core Rule Set
variable "enable_owasp_core" {
  description = "Enable OWASP Core Rule Set"
  type        = bool
  default     = true
}

variable "enable_owasp_paranoia_level" {
  description = "OWASP paranoia level (1-4, higher = stricter)"
  type        = number
  default     = 2
  
  validation {
    condition     = contains([1, 2, 3, 4], var.enable_owasp_paranoia_level)
    error_message = "Paranoia level must be between 1 and 4"
  }
}

variable "anomaly_threshold" {
  description = "Anomaly score threshold for blocking (2-10000)"
  type        = number
  default     = 25
  
  validation {
    condition     = var.anomaly_threshold >= 2 && var.anomaly_threshold <= 10000
    error_message = "Anomaly threshold must be between 2 and 10000"
  }
}

# Mode
variable "waf_mode" {
  description = "WAF mode: 'MONITOR' (log only) or 'BLOCK'"
  type        = string
  default     = "BLOCK"
  
  validation {
    condition     = contains(["MONITOR", "BLOCK"], var.waf_mode)
    error_message = "WAF mode must be MONITOR or BLOCK"
  }
}

# Exclusion rules (simplified)
variable "exclusion_rules" {
  description = "Rules to exclude certain paths from WAF inspection"
  type = list(object({
    name        = string
    description = optional(string, "")
    paths       = optional(list(string), [])
    rule_ids    = optional(list(string), [])
  }))
  default = []
}

# Labels
variable "labels" {
  description = "Labels for WAF resources"
  type        = map(string)
  default     = {}
}

# Legacy variables (kept for compatibility - not actively used)
variable "enable_sql_injection_protection" {
  description = "Enabled via OWASP core rules"
  type        = bool
  default     = true
}

variable "enable_xss_protection" {
  description = "Enabled via OWASP core rules"
  type        = bool
  default     = true
}

variable "enable_path_traversal_protection" {
  description = "Enabled via OWASP core rules"
  type        = bool
  default     = true
}

variable "enable_scanner_detection" {
  description = "Enabled via OWASP core rules"
  type        = bool
  default     = true
}

variable "enable_bot_protection" {
  description = "Bot protection (not in WAF profile)"
  type        = bool
  default     = false
}

variable "bot_protection_mode" {
  description = "Bot protection mode"
  type        = string
  default     = "MONITOR"
}

variable "enable_rate_limiting" {
  description = "Rate limiting (not in WAF profile - use SmartWeb Security instead)"
  type        = bool
  default     = false
}

variable "rate_limit_rules" {
  description = "Rate limiting rules"
  type = list(object({
    name        = string
    path        = optional(string, "/")
    requests_per_second = number
    burst       = number
    action      = string
  }))
  default = []
}

variable "ip_whitelist" {
  description = "IP whitelist (use SmartWeb Security rules instead)"
  type        = list(string)
  default     = []
}

variable "ip_blacklist" {
  description = "IP blacklist (use SmartWeb Security rules instead)"
  type        = list(string)
  default     = []
}

variable "geo_rules" {
  description = "Geo rules (use SmartWeb Security rules instead)"
  type = list(object({
    name        = string
    countries   = list(string)
    action      = string
    priority    = number
  }))
  default = []
}

variable "max_request_body_size" {
  description = "Max request body size in bytes"
  type        = number
  default     = 10485760
}

variable "max_url_length" {
  description = "Max URL length"
  type        = number
  default     = 2048
}

variable "enable_logging" {
  description = "Enable logging"
  type        = bool
  default     = true
}

variable "log_group_id" {
  description = "Log group ID for WAF logs"
  type        = string
  default     = null
}

variable "enable_ddos_protection" {
  description = "DDoS protection (separate service)"
  type        = bool
  default     = false
}

variable "custom_rules" {
  description = "Custom rules (use SmartWeb Security instead)"
  type = list(object({
    name      = string
    priority  = number
    action    = string
    condition = object({
      path   = optional(string)
      method = optional(string)
      ip     = optional(string)
    })
  }))
  default = []
}