# Data source for folder ID
data "yandex_resourcemanager_folder" "current" {
  folder_id = var.folder_id
}

# Construct origin path
locals {
  origin_path = var.origin_source
}

# CDN Resource
resource "yandex_cdn_resource" "main" {
  cname               = var.cname
  active              = var.active
  folder_id           = var.folder_id
  secondary_hostnames = var.secondary_hostnames
  
  origin_protocol = var.origin_protocol
  origin_group_id = yandex_cdn_origin_group.main.id
  ssl_certificate {
    type = var.ssl_certificate_type
    certificate_manager_id = var.ssl_certificate_id
  }
  
  labels = merge(var.labels, {
    environment = var.name_prefix
    service     = "cdn"
  })
}

# Origin Group
resource "yandex_cdn_origin_group" "main" {
  name       = "${var.name_prefix}-origin-group"
  folder_id  = var.folder_id
  
  origin {
    source  = local.origin_path
    enabled = true
    backup  = false
  }
  
  use_next = true
}

# DNS Record (if custom domain is provided)
resource "yandex_dns_recordset" "cdn_cname" {
  count = var.create_dns_record ? 1 : 0
  
  zone_id = var.dns_zone_id
  name    = var.dns_record_name != null ? var.dns_record_name : var.cname
  type    = "CNAME"
  ttl     = var.dns_ttl
  data    = [yandex_cdn_resource.main.cname]
  
  depends_on = [yandex_cdn_resource.main]
}