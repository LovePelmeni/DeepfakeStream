locals {
  auto_dns_subdomain = var.service_name != null && var.environment != null ? "${var.service_name}-${var.environment}" : var.service_name != null ? var.service_name : var.name

  dns_enabled = var.create_dns_record && var.dns_config != null
  dns_create_ptr = local.dns_enabled ? var.dns_config.create_ptr : false

  dns_record_name = local.dns_enabled ? (var.dns_config.subdomain != "" ? var.dns_config.subdomain : local.auto_dns_subdomain) : null
}

resource "yandex_vpc_address" "static_ip" {
  name                = var.name
  folder_id           = var.folder_id
  description         = var.description
  labels              = var.labels
  deletion_protection = var.deletion_protection

  external_ipv4_address {
    zone_id = var.zone_id
  }
}

data "yandex_dns_zone" "zone" {
  count       = local.dns_enabled ? 1 : 0
  dns_zone_id = var.dns_config.zone_id
  folder_id   = var.dns_zone_folder_id != null ? var.dns_zone_folder_id : var.folder_id
}

resource "yandex_dns_recordset" "a_record" {
  count   = local.dns_enabled ? 1 : 0
  zone_id = var.dns_config.zone_id
  name    = var.dns_config.subdomain != "" ? "${var.dns_config.subdomain}.${var.dns_config.domain}" : var.dns_config.domain
  type    = "A"
  ttl     = var.dns_config.ttl
  data    = [yandex_vpc_address.static_ip.external_ipv4_address[0].address]

  description = var.dns_config.description != "" ? var.dns_config.description : "A record for ${var.name} IP address"
}

resource "yandex_dns_recordset" "ptr_record" {
  count   = local.dns_create_ptr ? 1 : 0
  zone_id = var.dns_config.zone_id
  name    = join(".", slice(split(".", yandex_vpc_address.static_ip.external_ipv4_address[0].address), 0, 4))
  type    = "PTR"
  ttl     = var.dns_config.ptr_ttl
  data    = [var.dns_config.subdomain != "" ? "${var.dns_config.subdomain}.${var.dns_config.domain}" : var.dns_config.domain]

  description = "PTR record for ${var.name} IP address"
}