variable "folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "registry_name" {
  description = "Name of the container registry"
  type        = string
}

variable "labels" {
  description = "Labels to attach to the registry"
  type        = map(string)
  default     = {}
}

variable "services" {
  description = "Map of services that need repositories"
  type = map(object({
    description = optional(string, "")
  }))
  default = {}
}