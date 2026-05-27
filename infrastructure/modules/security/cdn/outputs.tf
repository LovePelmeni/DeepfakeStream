output "cdn_resource_id" {
  description = "CDN resource ID"
  value       = yandex_cdn_resource.main.id
}

output "cdn_resource_cname" {
  description = "CDN resource CNAME (the endpoint to point your domain to)"
  value       = yandex_cdn_resource.main.cname
}

output "cdn_resource_active" {
  description = "Whether the CDN resource is active"
  value       = yandex_cdn_resource.main.active
}

output "origin_group_id" {
  description = "Origin group ID"
  value       = yandex_cdn_origin_group.main.id
}

output "origin_group_name" {
  description = "Origin group name"
  value       = yandex_cdn_origin_group.main.name
}

output "origin_group_origins" {
  description = "List of origins in the origin group"
  value       = yandex_cdn_origin_group.main.origin[*].source
}

output "custom_domain" {
  description = "Custom domain (if configured)"
  value       = var.cname
}

output "cdn_endpoint" {
  description = "CDN endpoint URL"
  value       = "https://${yandex_cdn_resource.main.cname}"
}

output "cdn_endpoint_http" {
  description = "CDN endpoint URL (HTTP)"
  value       = "http://${yandex_cdn_resource.main.cname}"
}

output "has_ssl_certificate" {
  description = "Whether SSL certificate is configured"
  value       = var.ssl_certificate_type != "not_used"
}

output "ssl_certificate_type" {
  description = "SSL certificate type"
  value       = var.ssl_certificate_type
}

output "dns_record_created" {
  description = "Whether DNS record was created"
  value       = var.create_dns_record
}