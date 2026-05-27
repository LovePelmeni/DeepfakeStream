# PostgreSQL Database Variables
variable "pg_name" {
  description = "PostgreSQL cluster name"
  type        = string
  default     = "postgres-cluster"
}

variable "pg_environment" {
  description = "PostgreSQL environment (prod/nonprod)"
  type        = string
  default     = "nonprod"
}

variable "pg_hosts_definition" {
  description = "PostgreSQL hosts configuration"
  type        = any
  default     = {
    zone_id = "ru-central1-a"
    size    = 1
  }
}

variable "pg_resource_preset_id" {
  description = "PostgreSQL resource preset ID"
  type        = string
  default     = "s2.micro"
}

variable "pg_disk_type_id" {
  description = "PostgreSQL disk type"
  type        = string
  default     = "network-ssd"
}

variable "pg_disk_size" {
  description = "PostgreSQL disk size (GB)"
  type        = number
  default     = 16
}

variable "pg_version" {
  description = "PostgreSQL version"
  type        = string
  default     = "15"
}

variable "pg_settings_options" {
  description = "PostgreSQL settings options"
  type        = list(map(string))
  default     = []
}

variable "pg_maintenance_window" {
  description = "PostgreSQL maintenance window"
  type = object({
    type = string
    day = optional(string)
    hour = optional(string)
  })
  default = {
    type = "ANYTIME"
  }
}

variable "pg_databases" {
  description = "PostgreSQL databases to create"
  type = list(object({
    name      = string
    owner     = string
    extensions = optional(list(string), [])
    lc_collate = optional(string)
    lc_ctype   = optional(string)
  }))
  default = []
}

variable "pg_owners" {
  description = "PostgreSQL database owners"
  type = list(object({
    name       = string
    password   = string
  }))
  default = []
  sensitive = true
}

variable "pg_users" {
  description = "PostgreSQL users"
  type = list(object({
    name       = string
    password   = string
    database_name = string
    roles      = optional(list(string), [])
  }))
  default = []
  sensitive = true
}

variable "pg_deletion_protection" {
  description = "Prevent accidental deletion of PostgreSQL cluster"
  type        = bool
  default     = true
}
