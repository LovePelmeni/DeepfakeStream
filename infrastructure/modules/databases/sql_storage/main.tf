# PRODUCTION-READY Managed PostgreSQL for Vegamaps GIS
module "pgsql" {
  source = "github.com/terraform-yc-modules/terraform-yc-postgresql"

  # === BASIC CONFIG ===
  name        = var.pg_name
  environment = var.pg_environment

  # === NETWORK ===
  network_id          = module.vpc.net.id
  security_groups_ids = [yandex_vpc_security_group.backend_server_sg.id]
  subnet_ids          = [yandex_vpc_subnet.private_sb.id]

  # === CLUSTER NODES ===
  hosts_definition = var.pg_hosts_definition

  # === RESOURCES ===
  resource_preset_id = var.pg_resource_preset_id
  disk_type_id       = var.pg_disk_type_id
  disk_size          = var.pg_disk_size

  # === VERSION & PERFORMANCE ===
  version          = var.pg_version
  settings_options = var.pg_settings_options

  # === MAINTENANCE ===
  maintenance_window = var.pg_maintenance_window

  # === DATABASES & ACCESS ===
  databases = var.pg_databases
  owners    = var.pg_owners
  users     = var.pg_users

  # === SAFETY ===
  deletion_protection = var.pg_deletion_protection
}

# Secure random passwords
resource "random_password" "pg_admin_password" {
  length  = 32
  special = true
}

resource "random_password" "pg_app_password" {
  length  = 32
  special = true
}

resource "random_password" "pg_readonly_password" {
  length  = 32
  special = true
}
