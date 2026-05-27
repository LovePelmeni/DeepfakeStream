variable "net_name" {
  description = "VPC network name"
  type        = string
}

variable "yandex_cloud_folder_id" {
  description = "Yandex Cloud folder ID"
  type        = string
}

variable "yandex_cloud_id" {
  description = "Yandex Cloud organization ID" 
  type        = string
}

variable "net_private_subnets" {
  description = "Private subnets configuration"
  type = list(object({
    zone  = string
    cidr  = string
  }))
}

variable "net_public_subnets" {
  description = "Public subnets configuration"
  type = list(object({
    zone  = string
    cidr  = string
  }))
}

variable "net_create_vpc" {
  description = "Create VPC (true/false)"
  type        = bool
  default     = true
}
