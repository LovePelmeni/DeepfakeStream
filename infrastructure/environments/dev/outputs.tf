
# ============================================================
# OUTPUTS
# ============================================================
output "alb_ip" {
  description = "ALB external IP address"
  value       = module.alb_static_ip.ip_address
}

output "bastion_ip" {
  description = "Bastion external IP address"
  value       = module.bastion_static_ip.ip_address
}

output "observability_ip" {
  description = "Observability server external IP address"
  value       = module.observability_static_ip.ip_address
}

output "database_hosts" {
  description = "PostgreSQL cluster hosts"
  value       = yandex_mdb_postgresql_cluster.shared.host[*].fqdn
}

output "database_password" {
  description = "Database password"
  value       = random_password.db_password.result
  sensitive   = true
}

output "teleport_auth_token" {
  description = "Teleport auth token"
  value       = random_password.teleport_token.result
  sensitive   = true
}

output "registry_id" {
  description = "Container registry ID"
  value       = yandex_container_registry.prod_registry.id
}

output "s3_bucket_name" {
  description = "S3 bucket name"
  value       = module.s3_bucket.bucket_name
}

output "grafana_url" {
  description = "Grafana UI URL"
  value       = "http://${module.observability_static_ip.ip_address}:3000"
}

output "prometheus_url" {
  description = "Prometheus UI URL"
  value       = "http://${module.observability_static_ip.ip_address}:9090"
}