# ============================================
# REQUIRED VARIABLES (used in main.tf)
# ============================================

variable "folder_id" {
  description = "Yandex Cloud folder ID where service accounts will be created"
  type        = string
}

variable "name_prefix" {
  description = "Prefix for naming service accounts"
  type        = string
  default     = "vegamaps"
}

# ============================================
# FEATURE FLAGS (used in main.tf with count)
# ============================================

variable "enable_consul_backups" {
  description = "Enable creation of Consul backup service account"
  type        = bool
  default     = true
}

variable "enable_consul_monitoring" {
  description = "Enable creation of Consul monitoring service account"
  type        = bool
  default     = true
}

variable "create_registry_admin" {
  description = "Create registry admin service account"
  type        = bool
  default     = false
}

# ============================================
# CONTAINER REGISTRY (used in for_each)
# ============================================

variable "service_registry_pushers" {
  description = "Map of services that need registry push permissions"
  type        = map(string)
  default = {
    backend          = "backend"
    frontend         = "frontend"
    ml               = "ml"
    websocket_collab = "websocket_collab"
  }
}