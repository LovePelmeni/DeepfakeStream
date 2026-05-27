output "ip_id" {
  description = "ID of the static IP address"
  value       = yandex_vpc_address.static_ip.id
}

output "ip_address" {
  description = "The static IP address value"
  value       = yandex_vpc_address.static_ip.external_ipv4_address[0].address
}

output "ip_zone" {
  description = "Zone where the IP address is located"
  value       = yandex_vpc_address.static_ip.external_ipv4_address[0].zone_id
}

output "ip_name" {
  description = "Name of the static IP address"
  value       = yandex_vpc_address.static_ip.name
}

output "ip_created_at" {
  description = "Creation timestamp of the static IP"
  value       = yandex_vpc_address.static_ip.created_at
}

output "dns_record_name" {
  description = "DNS record name created for this IP"
  value       = local.dns_enabled && length(yandex_dns_recordset.a_record) > 0 ? yandex_dns_recordset.a_record[0].name : null
}

output "dns_record_fqdn" {
  description = "Fully qualified domain name for the IP"
  value       = local.dns_enabled && length(yandex_dns_recordset.a_record) > 0 ? yandex_dns_recordset.a_record[0].name : null
}

output "dns_zone_id" {
  description = "DNS zone ID used for the record"
  value       = local.dns_enabled ? var.dns_config.zone_id : null
}

output "full_endpoint" {
  description = "Complete endpoint URL (if DNS configured)"
  value       = local.dns_enabled && length(yandex_dns_recordset.a_record) > 0 ? yandex_dns_recordset.a_record[0].name : yandex_vpc_address.static_ip.external_ipv4_address[0].address
}