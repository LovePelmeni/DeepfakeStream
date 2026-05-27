locals {
  prefix = var.name_prefix != "" ? "${var.name_prefix}-" : ""
}

# ------------------------------------------------------------------------------
# Security & Incident Response Accounts
# ------------------------------------------------------------------------------

resource "yandex_iam_service_account" "security_scanner" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-security-scanner"
  description = "Service account for security scanning (read-only)"
}

resource "yandex_resourcemanager_folder_iam_member" "scanner_roles" {
  for_each = toset([
    "logging.viewer",
    "dspm.worker",
    "iam.viewer",
    "compute.viewer",
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.security_scanner.id}"
}

resource "yandex_iam_service_account" "bucket_fixer" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-bucket-fixer"
  description = "Service account to fix public bucket permissions"
}

resource "yandex_resourcemanager_folder_iam_member" "bucket_fixer" {
  folder_id = var.folder_id
  role      = "storage.editor"
  member    = "serviceAccount:${yandex_iam_service_account.bucket_fixer.id}"
}

resource "yandex_iam_service_account" "vm_locker" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-vm-locker"
  description = "Service account to lock down VMs (security group changes)"
}

resource "yandex_resourcemanager_folder_iam_member" "vm_locker" {
  folder_id = var.folder_id
  role      = "compute.editor"
  member    = "serviceAccount:${yandex_iam_service_account.vm_locker.id}"
}

resource "yandex_iam_service_account" "notifier" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-notifier"
  description = "Service account to send notifications based on logs"
}

resource "yandex_resourcemanager_folder_iam_member" "notifier" {
  folder_id = var.folder_id
  role      = "logging.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.notifier.id}"
}

resource "yandex_iam_service_account" "tracker" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-incident-tracker"
  description = "Service account for incident tracking"
}

resource "yandex_iam_service_account" "response_orchestrator" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-response-orchestrator"
  description = "Service account with broad permissions for incident response"
}

resource "yandex_resourcemanager_folder_iam_member" "response_orchestrator" {
  for_each = toset([
    "storage.editor",
    "compute.editor",
    "vpc.admin",
    "logging.viewer",
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.response_orchestrator.id}"
}

resource "yandex_iam_service_account" "security_deck_scanner" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-deck-scanner"
  description = "Service account for Security Deck scanning (CSPM, DSPM, CIEM)"
}

resource "yandex_resourcemanager_folder_iam_member" "deck_scanner" {
  for_each = toset([
    "compute.viewer",
    "storage.viewer",
    "vpc.viewer",
    "iam.viewer",
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.security_deck_scanner.id}"
}

# Observability Service Account
resource "yandex_iam_service_account" "observability" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-observability"
  description = "Service account for observability server (Grafana, Prometheus, Wazuh)"
}

resource "yandex_resourcemanager_folder_iam_member" "observability_roles" {
  for_each = toset([
    "monitoring.viewer",           # View monitoring data
    "logging.reader",              # Read logs
    "compute.viewer",              # View VM metrics
    "storage.viewer",              # View bucket stats
    "kms.keys.encrypterDecrypter", # Encrypt/decrypt with KMS
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.observability.id}"
}

# ============================================
# CONSUL SERVICE MESH IAM ACCOUNTS
# ============================================

# Main Consul server service account
resource "yandex_iam_service_account" "consul_server" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-consul-server"
  description = "Service account for Consul server cluster"
}

resource "yandex_resourcemanager_folder_iam_member" "consul_server_roles" {
  for_each = toset([
    "compute.viewer",           # Read VM metadata and instances
    "vpc.viewer",              # Read network configuration
    "load-balancer.viewer",    # Read load balancer configs
    "dns.editor",              # Create/update DNS records for service discovery
    "kms.keys.encrypterDecrypter", # Encrypt gossip key and certificates
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.consul_server.id}"
}

# Consul sidecar service account (for application instances)
resource "yandex_iam_service_account" "consul_sidecar" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-consul-sidecar"
  description = "Service account for Consul sidecar proxies on application VMs"
}

resource "yandex_resourcemanager_folder_iam_member" "consul_sidecar_roles" {
  for_each = toset([
    "compute.viewer",           # Read service metadata
    "vpc.viewer",              # Read network configs
    "load-balancer.viewer",    # Read LB configuration
  ])

  folder_id = var.folder_id
  role      = each.key
  member    = "serviceAccount:${yandex_iam_service_account.consul_sidecar.id}"
}

# Consul backup service account (for S3 snapshots)
resource "yandex_iam_service_account" "consul_backup" {
  count = var.enable_consul_backups ? 1 : 0
  
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-consul-backup"
  description = "Service account for Consul state backups to S3"
}

resource "yandex_resourcemanager_folder_iam_member" "consul_backup_roles" {
  count = var.enable_consul_backups ? 1 : 0
  
  folder_id = var.folder_id
  role      = "storage.editor"
  member    = "serviceAccount:${yandex_iam_service_account.consul_backup[0].id}"
}

# Consul monitoring service account
resource "yandex_iam_service_account" "consul_monitoring" {
  count = var.enable_consul_monitoring ? 1 : 0
  
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-consul-monitoring"
  description = "Service account for Consul metrics scraping and monitoring"
}

resource "yandex_resourcemanager_folder_iam_member" "consul_monitoring_roles" {
  count = var.enable_consul_monitoring ? 1 : 0
  
  folder_id = var.folder_id
  role      = "monitoring.editor"
  member    = "serviceAccount:${yandex_iam_service_account.consul_monitoring[0].id}"
}

resource "yandex_resourcemanager_folder_iam_member" "consul_monitoring_roles_compute" {
  count = var.enable_consul_monitoring ? 1 : 0
  
  folder_id = var.folder_id
  role      = "compute.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.consul_monitoring[0].id}"
}

# Static access keys for Consul backup (stored in Lockbox)
resource "yandex_iam_service_account_static_access_key" "consul_backup" {
  count = var.enable_consul_backups ? 1 : 0
  
  service_account_id = yandex_iam_service_account.consul_backup[0].id
  description        = "Access key for Consul S3 backups"
}

# ------------------------------------------------------------------------------
# Application Service Accounts
# ------------------------------------------------------------------------------

resource "yandex_iam_service_account" "backend" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-backend"
  description = "Service account for backend compute instance"
}

resource "yandex_iam_service_account" "ml" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-ml"
  description = "Service account for ML compute instance (GPU)"
}

resource "yandex_iam_service_account" "frontend" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-frontend"
  description = "Service account for frontend compute instance"
}

resource "yandex_iam_service_account" "websocket_collab" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-ws-collab"
  description = "Service account for WebSocket collaboration instance"
}

resource "yandex_iam_service_account" "s3_access" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-s3-access"
  description = "Service account for applications to access S3 bucket"
}

# ------------------------------------------------------------------------------
# Infrastructure Management Accounts
# ------------------------------------------------------------------------------

resource "yandex_iam_service_account" "secrets_manager" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-secrets-manager"
  description = "Service account to manage KMS keys and Lockbox secrets"
}

resource "yandex_resourcemanager_folder_iam_member" "secrets_manager_kms" {
  folder_id = var.folder_id
  role      = "kms.admin"
  member    = "serviceAccount:${yandex_iam_service_account.secrets_manager.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "secrets_manager_lockbox" {
  folder_id = var.folder_id
  role      = "lockbox.admin"
  member    = "serviceAccount:${yandex_iam_service_account.secrets_manager.id}"
}

resource "yandex_iam_service_account" "network_manager" {
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-network-manager"
  description = "Service account to manage VPC networking resources"
}

resource "yandex_resourcemanager_folder_iam_member" "network_manager_vpc" {
  folder_id = var.folder_id
  role      = "vpc.admin"
  member    = "serviceAccount:${yandex_iam_service_account.network_manager.id}"
}

# ------------------------------------------------------------------------------
# Container Registry & Docker Registry IAM
# ------------------------------------------------------------------------------

resource "yandex_iam_service_account" "registry_admin" {
  count       = var.create_registry_admin ? 1 : 0
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-registry-admin"
  description = "Service account for container registry administration"
}

resource "yandex_resourcemanager_folder_iam_member" "registry_admin" {
  count     = var.create_registry_admin ? 1 : 0
  folder_id = var.folder_id
  role      = "container-registry.admin"
  member    = "serviceAccount:${yandex_iam_service_account.registry_admin[0].id}"
}

resource "yandex_iam_service_account" "registry_pusher" {
  for_each    = var.service_registry_pushers
  folder_id   = var.folder_id
  name        = "${local.prefix}vegamaps-${replace(each.key, "_", "-")}-registry-pusher"
  description = "Service account for ${each.key} to push images to container registry"
}

resource "yandex_resourcemanager_folder_iam_member" "registry_pusher_permissions" {
  for_each  = yandex_iam_service_account.registry_pusher
  folder_id = var.folder_id
  role      = "container-registry.images.pusher"
  member    = "serviceAccount:${each.value.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "app_puller_permissions" {
  for_each = {
    backend          = yandex_iam_service_account.backend.id
    frontend         = yandex_iam_service_account.frontend.id
    ml               = yandex_iam_service_account.ml.id
    websocket-collab = yandex_iam_service_account.websocket_collab.id
  }

  folder_id = var.folder_id
  role      = "container-registry.images.puller"
  member    = "serviceAccount:${each.value}"
}

resource "yandex_resourcemanager_folder_iam_member" "response_orchestrator_registry" {
  folder_id = var.folder_id
  role      = "container-registry.editor"
  member    = "serviceAccount:${yandex_iam_service_account.response_orchestrator.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "security_scanner_registry" {
  folder_id = var.folder_id
  role      = "container-registry.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.security_scanner.id}"
}

resource "yandex_resourcemanager_folder_iam_member" "deck_scanner_registry" {
  folder_id = var.folder_id
  role      = "container-registry.viewer"
  member    = "serviceAccount:${yandex_iam_service_account.security_deck_scanner.id}"
}