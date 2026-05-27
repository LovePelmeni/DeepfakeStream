terraform {
  required_providers {
    yandex = {
      source  = "yandex-cloud/yandex"
      version = "~> 0.188"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  required_version = ">= 0.13"

  backend "s3" {
    bucket                      = "vegamaps-infra-state-bucket"
    key                         = "prod/terraform.tfstate"
    region                      = "ru-central1"
    endpoint                    = "storage.yandexcloud.net"
    skip_credentials_validation = true
    skip_region_validation      = true
    skip_metadata_api_check     = true
    force_path_style            = true
  }
}

provider "yandex" {
  cloud_id                 = var.yandex_cloud_id
  folder_id                = var.yandex_cloud_folder_id
  zone                     = var.yandex_cloud_zone
  service_account_key_file = var.yandex_service_account_key_file
}

# ============================================================
# LOCALS
# ============================================================
locals {
  teleport_deb_base64 = filebase64("${path.module}/../../modules/security/bastion/files/linux/teleport_18.7.0_amd64.deb")
  
  # Helper for memory bytes
  memory_to_bytes = { for k, v in merge(local.free_tier, local.paid_tier) : k => v.memory_gb * 1073741824 if can(v.memory_gb) }
}

# ============================================================
# RANDOM PASSWORDS
# ============================================================
resource "random_password" "teleport_token" {
  length  = 32
  special = false
}

resource "random_password" "db_password" {
  length  = 24
  special = true
}

# ============================================================
# VPC NETWORK
# ============================================================
module "vpc" {
  source = "../../modules/core/vpc"

  net_name               = "prod-vpc"
  yandex_cloud_id        = var.yandex_cloud_id
  yandex_cloud_folder_id = var.yandex_cloud_folder_id
  net_create_vpc         = true

  net_private_subnets = [
    { zone = "ru-central1-a", cidr = "10.0.20.0/24" },
    { zone = "ru-central1-b", cidr = "10.0.21.0/24" },
  ]

  net_public_subnets = [
    { zone = "ru-central1-a", cidr = "10.0.10.0/24" }
  ]
}

# ============================================================
# IAM SERVICE ACCOUNTS
# ============================================================
module "iam_accounts" {
  source = "../../modules/admin/iam_accounts"

  folder_id   = var.yandex_cloud_folder_id
  name_prefix = local.environment
}

# ============================================================
# STATIC IPS
# ============================================================
module "alb_static_ip" {
  source = "../../modules/core/static_ip"
  name        = "${local.environment}-alb-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for ${local.environment} ALB"
  labels      = { environment = local.environment, service = "alb" }
  deletion_protection = true
}

module "bastion_static_ip" {
  source = "../../modules/core/static_ip"
  name        = "${local.environment}-bastion-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for ${local.environment} bastion"
  labels      = { environment = local.environment, service = "bastion" }
  deletion_protection = true
}

module "observability_static_ip" {
  source = "../../modules/core/static_ip"
  name        = "${local.environment}-observability-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for ${local.environment} observability server"
  labels      = { environment = local.environment, service = "observability" }
  deletion_protection = true
}

# ============================================================
# SECURITY GROUPS
# ============================================================
resource "yandex_vpc_security_group" "alb_security_group" {
  name        = var.alb_sg_name
  description = "Application Load Balancer security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "HTTP from internet"
    port           = 80
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    protocol       = "TCP"
    description    = "HTTPS from internet"
    port           = 443
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound to private subnets"
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  labels = { environment = local.environment, service = "alb" }
  lifecycle { prevent_destroy = true }
}

resource "yandex_vpc_security_group" "observability_security_group" {
  name        = var.observability_sg_name
  description = "Observability server security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  dynamic "ingress" {
    for_each = {
      "3000" = "Grafana UI"
      "9090" = "Prometheus UI"
      "5000" = "MLflow UI"
    }
    content {
      protocol       = "TCP"
      description    = ingress.value
      port           = ingress.key
      v4_cidr_blocks = ["0.0.0.0/0"]
    }
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    v4_cidr_blocks = var.bastion_admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "observability" }
}

resource "yandex_vpc_security_group" "backend_server_security_group" {
  name        = var.backend_server_sg_name
  description = "Backend server security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "API from ALB"
    port           = 8080
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    v4_cidr_blocks = var.bastion_metadata_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "backend" }
}

resource "yandex_vpc_security_group" "frontend_security_group" {
  name        = var.frontend_server_sg_name
  description = "Frontend server security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "HTTP from ALB"
    port           = 80
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    v4_cidr_blocks = var.bastion_metadata_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "frontend" }
}

resource "yandex_vpc_security_group" "ml_service_security_group" {
  name        = var.ml_service_sg_name
  description = "ML service security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "MLflow tracking from ALB"
    port           = 5000
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Model serving from ALB"
    port           = 8080
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    v4_cidr_blocks = var.bastion_metadata_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "ml" }
}

resource "yandex_vpc_security_group" "websocket_collab_security_group" {
  name        = var.websocket_collab_sg_name
  description = "Websocket collaboration server security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "WebSocket TLS from ALB"
    port           = 443
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "WebSocket from ALB"
    port           = 80
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    v4_cidr_blocks = var.bastion_metadata_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "websocket" }
}

resource "yandex_vpc_security_group" "database_security_group" {
  name        = var.database_sg_name
  description = "Database security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol          = "TCP"
    description       = "Postgres from backend servers"
    port              = 5432
    security_group_id = yandex_vpc_security_group.backend_server_security_group.id
  }
  ingress {
    protocol          = "TCP"
    description       = "Postgres from ML servers"
    port              = 5432
    security_group_id = yandex_vpc_security_group.ml_service_security_group.id
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "database" }
}

resource "yandex_vpc_security_group" "bastion_security_group" {
  name        = var.bastion_sg_name
  description = "Bastion host security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "Admin SSH access"
    port           = 22
    v4_cidr_blocks = var.bastion_admin_ip_cidr
  }
  ingress {
    protocol       = "TCP"
    description    = "Teleport Auth from nodes"
    port           = 3025
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Teleport SSH Proxy from admin"
    port           = 3022
    v4_cidr_blocks = var.bastion_admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "bastion" }
}

# ============================================================
# SHARED POSTGRESQL DATABASE
# ============================================================
resource "yandex_mdb_postgresql_cluster" "shared" {
  name        = "vegamaps-shared-db"
  folder_id   = var.yandex_cloud_folder_id
  environment = "PRODUCTION"
  network_id  = module.vpc.vpc_id

  config {
    version = local.global_config.postgresql.version
    resources {
      resource_preset_id = local.global_config.postgresql.preset_id
      disk_size          = local.global_config.postgresql.disk_size
      disk_type_id       = local.global_config.defaults.boot_disk_type
    }
    postgresql_config = {
      max_connections            = local.global_config.postgresql.max_connections
      enable_parallel_hash       = true
      log_min_duration_statement = 5000
    }
  }

  host {
    zone      = "ru-central1-a"
    subnet_id = module.vpc.private_subnet_ids["ru-central1-a"]
    assign_public_ip = false
  }
  host {
    zone      = "ru-central1-b"
    subnet_id = module.vpc.private_subnet_ids["ru-central1-b"]
    assign_public_ip = false
  }

  security_group_ids = [yandex_vpc_security_group.database_security_group.id]
  labels = { environment = local.environment, service = "postgres" }
  deletion_protection = true
}

resource "yandex_mdb_postgresql_user" "vegamaps_user" {
  cluster_id = yandex_mdb_postgresql_cluster.shared.id
  name       = "vegamaps_user"
  password   = random_password.db_password.result
  conn_limit = 100
}

resource "yandex_mdb_postgresql_database" "vegamaps" {
  cluster_id = yandex_mdb_postgresql_cluster.shared.id
  name       = "vegamaps"
  owner      = yandex_mdb_postgresql_user.vegamaps_user.name
  depends_on = [yandex_mdb_postgresql_user.vegamaps_user]
}

# ============================================================
# BASTION HOST
# ============================================================
module "bastion" {
  source = "../../modules/security/bastion"

  bastion_env_name           = local.environment
  bastion_folder_id          = var.yandex_cloud_folder_id
  bastion_availability_zone  = "ru-central1-a"
  bastion_platform_id        = "standard-v3"
  bastion_service_account_id = module.iam_accounts.response_orchestrator_sa_id
  bastion_subnet_id          = module.vpc.public_subnet_ids["ru-central1-a"]
  bastion_security_group_ids = [yandex_vpc_security_group.bastion_security_group.id]
  bastion_os_image           = var.bastion_os_image
  bastion_ssh_keys           = "ubuntu:${var.bastion_ssh_public_key}"
  bastion_number_of_cores    = local.global_config.bastion.cores
  bastion_ram_gb             = local.global_config.bastion.memory_gb
  bastion_boot_disk_size     = local.global_config.bastion.disk_size

  bastion_static_ip_address = module.bastion_static_ip.ip_address
  teleport_auth_token       = random_password.teleport_token.result
  teleport_version          = "18.7.0"
}

# ============================================================
# OBSERVABILITY SERVER
# ============================================================
module "observability" {
  source = "../../modules/observability"

  observability_env_name  = local.environment
  observability_folder_id = var.yandex_cloud_folder_id
  observability_zone      = "ru-central1-a"
  observability_service_account_id = module.iam_accounts.observability_sa_id
  observability_subnet_id = module.vpc.public_subnet_ids["ru-central1-a"]
  observability_security_group_ids = [yandex_vpc_security_group.observability_security_group.id]

  observability_cores          = local.global_config.observability.cores
  observability_core_fraction  = local.global_config.defaults.core_fraction
  observability_ram_gb         = local.global_config.observability.memory_gb

  observability_boot_disk_size = local.global_config.observability.boot_disk_size
  observability_boot_disk_type = local.global_config.defaults.boot_disk_type

  observability_data_disk_enabled = local.global_config.observability.data_disk_enabled
  observability_data_disk_size    = local.global_config.observability.data_disk_size
  observability_data_disk_type    = local.global_config.defaults.boot_disk_type

  observability_static_ip_address = module.observability_static_ip.ip_address
  observability_preemptible       = local.global_config.defaults.preemptible

  observability_os_image       = "ubuntu-2204-lts"
  observability_ssh_user       = "ubuntu"
  observability_ssh_public_key = var.observability_ssh_public_key

  observability_enable_monitoring = true
  observability_services          = local.global_config.observability.services

  install_wazuh              = false
  observability_extra_labels = {
    environment = local.environment
    cost_center = "platform"
    team        = "devops"
  }
}

# ============================================================
# S3 BUCKET
# ============================================================
module "s3_bucket" {
  source = "../../modules/databases/s3_storage"

  s3_yandex_cloud_folder_id = var.yandex_cloud_folder_id
  s3_bucket_name            = var.s3_bucket_name
  s3_yandex_cloud_id        = var.yandex_cloud_id
}

# ============================================================
# CONTAINER REGISTRY
# ============================================================
resource "yandex_container_registry" "prod_registry" {
  name      = var.registry_name
  folder_id = var.yandex_cloud_folder_id
  labels    = var.registry_labels
}

resource "yandex_container_repository" "backend" {
  name = "${yandex_container_registry.prod_registry.id}/backend"
}
resource "yandex_container_repository" "frontend" {
  name = "${yandex_container_registry.prod_registry.id}/frontend"
}
resource "yandex_container_repository" "ml" {
  name = "${yandex_container_registry.prod_registry.id}/ml"
}
resource "yandex_container_repository" "websocket_collab" {
  name = "${yandex_container_registry.prod_registry.id}/websocket-collab"
}

# ============================================================
# KMS & LOCKBOX
# ============================================================
module "secrets" {
  source = "../../modules/admin/secrets"

  folder_id              = var.yandex_cloud_folder_id
  name_prefix            = local.environment
  secrets_manager_sa_id  = module.iam_accounts.secrets_manager_sa_id

  kms_key_name           = "${local.environment}-kms-key"
  kms_description        = "${local.environment} encryption key"
  kms_rotation_period    = "720h"
  kms_deletion_protection = true

  lockbox_secret_name      = "${local.environment}-secrets"
  lockbox_description      = "${local.environment} secrets"
  lockbox_deletion_protection = true

  lockbox_version_payload = {
    DB_PASSWORD = random_password.db_password.result
  }

  labels = {
    environment = local.environment
    managed_by  = "terraform"
  }
}

# ============================================================
# FREE TIER SERVICES
# ============================================================
module "backend_free" {
  source = "../../modules/services/backend_service"

  backend_env_name           = "free"
  backend_folder_id          = var.yandex_cloud_folder_id
  backend_service_account_id = module.iam_accounts.backend_sa_id
  backend_allocation_zones   = [local.free_tier.zone]
  backend_subnet_id          = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  backend_security_group_ids = [yandex_vpc_security_group.backend_server_security_group.id]
  backend_os_image           = var.backend_os_image
  backend_ssh_public_key     = var.backend_ssh_public_key
  backend_number_of_cores    = local.free_tier.backend.cores
  backend_ram_memory         = local.free_tier.backend.memory_gb * 1073741824
  backend_boot_disk_size     = local.free_tier.backend.disk_size
  backend_assign_public_ip   = local.global_config.defaults.assign_public_ip
  instance_count             = local.free_tier.backend.instance_count

  db_connection_string = "postgresql://vegamaps_user:${random_password.db_password.result}@${yandex_mdb_postgresql_cluster.shared.host[0].fqdn}:6432/vegamaps"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "frontend_free" {
  source = "../../modules/services/frontend_service"

  name_prefix          = "frontend-free-${local.environment}"
  folder_id            = var.yandex_cloud_folder_id
  zone                 = local.free_tier.zone
  subnet_id            = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  service_account_id   = module.iam_accounts.frontend_sa_id
  security_group_ids   = [yandex_vpc_security_group.frontend_security_group.id]
  image_id             = var.frontend_os_image
  ssh_public_key       = var.frontend_ssh_public_key
  cores                = local.free_tier.frontend.cores
  memory               = local.free_tier.frontend.memory_gb * 1073741824
  boot_disk_size       = local.free_tier.frontend.disk_size
  assign_public_ip     = local.global_config.defaults.assign_public_ip
  environment          = "free"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "ml_free" {
  source = "../../modules/services/ml_service"

  name_prefix          = "ml-free-${local.environment}"
  folder_id            = var.yandex_cloud_folder_id
  zone                 = local.free_tier.zone
  subnet_id            = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  service_account_id   = module.iam_accounts.ml_sa_id
  security_group_ids   = [yandex_vpc_security_group.ml_service_security_group.id]
  platform_id          = local.free_tier.ml.platform
  cores                = local.free_tier.ml.cores
  memory_gb            = local.free_tier.ml.memory_gb
  gpus                 = local.free_tier.ml.gpus
  boot_disk_size       = local.free_tier.ml.disk_size
  image_id             = var.ml_os_image
  ssh_public_key       = var.ml_ssh_public_key
  assign_public_ip     = local.global_config.defaults.assign_public_ip
  environment          = "free"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "websocket_free" {
  source = "../../modules/services/websocket_collab_server"

  ws_collab_name                 = "ws-free-${local.environment}"
  ws_collab_folder_id            = var.yandex_cloud_folder_id
  ws_collab_service_account_id   = module.iam_accounts.websocket_collab_sa_id
  ws_collab_zone                 = local.free_tier.zone
  ws_collab_subnet_id            = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  ws_collab_security_group_ids   = [yandex_vpc_security_group.websocket_collab_security_group.id]
  ws_collab_image_id             = var.ws_os_image
  ws_collab_ssh_public_key       = var.ws_ssh_public_key
  ws_collab_cores                = local.free_tier.websocket.cores
  ws_collab_memory_gb            = local.free_tier.websocket.memory_gb
  ws_collab_disk_size_gb         = local.free_tier.websocket.disk_size
  ws_collab_enable_nat           = local.global_config.defaults.assign_public_ip
  ws_collab_environment          = "free"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

# ============================================================
# PAID TIER SERVICES
# ============================================================
module "backend_paid" {
  source = "../../modules/services/backend_service"

  backend_env_name           = "paid"
  backend_folder_id          = var.yandex_cloud_folder_id
  backend_service_account_id = module.iam_accounts.backend_sa_id
  backend_allocation_zones   = [local.paid_tier.zone]
  backend_subnet_id          = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  backend_security_group_ids = [yandex_vpc_security_group.backend_server_security_group.id]
  backend_os_image           = var.backend_os_image
  backend_ssh_public_key     = var.backend_ssh_public_key
  backend_number_of_cores    = local.paid_tier.backend.cores
  backend_ram_memory         = local.paid_tier.backend.memory_gb * 1073741824
  backend_boot_disk_size     = local.paid_tier.backend.disk_size
  backend_assign_public_ip   = local.global_config.defaults.assign_public_ip
  instance_count             = local.paid_tier.backend.instance_count

  db_connection_string = "postgresql://vegamaps_user:${random_password.db_password.result}@${yandex_mdb_postgresql_cluster.shared.host[0].fqdn}:6432/vegamaps"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "frontend_paid" {
  source = "../../modules/services/frontend_service"

  name_prefix          = "frontend-paid-${local.environment}"
  folder_id            = var.yandex_cloud_folder_id
  zone                 = local.paid_tier.zone
  subnet_id            = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  service_account_id   = module.iam_accounts.frontend_sa_id
  security_group_ids   = [yandex_vpc_security_group.frontend_security_group.id]
  image_id             = var.frontend_os_image
  ssh_public_key       = var.frontend_ssh_public_key
  cores                = local.paid_tier.frontend.cores
  memory               = local.paid_tier.frontend.memory_gb * 1073741824
  boot_disk_size       = local.paid_tier.frontend.disk_size
  assign_public_ip     = local.global_config.defaults.assign_public_ip
  environment          = "paid"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "ml_paid" {
  source = "../../modules/services/ml_service"

  name_prefix          = "ml-paid-${local.environment}"
  folder_id            = var.yandex_cloud_folder_id
  zone                 = local.paid_tier.zone
  subnet_id            = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  service_account_id   = module.iam_accounts.ml_sa_id
  security_group_ids   = [yandex_vpc_security_group.ml_service_security_group.id]
  platform_id          = local.paid_tier.ml.platform
  cores                = local.paid_tier.ml.cores
  memory_gb            = local.paid_tier.ml.memory_gb
  gpus                 = local.paid_tier.ml.gpus
  boot_disk_size       = local.paid_tier.ml.disk_size
  image_id             = var.ml_os_image
  ssh_public_key       = var.ml_ssh_public_key
  assign_public_ip     = local.global_config.defaults.assign_public_ip
  environment          = "paid"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}

module "websocket_paid" {
  source = "../../modules/services/websocket_collab_server"

  ws_collab_name                 = "ws-paid-${local.environment}"
  ws_collab_folder_id            = var.yandex_cloud_folder_id
  ws_collab_service_account_id   = module.iam_accounts.websocket_collab_sa_id
  ws_collab_zone                 = local.paid_tier.zone
  ws_collab_subnet_id            = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  ws_collab_security_group_ids   = [yandex_vpc_security_group.websocket_collab_security_group.id]
  ws_collab_image_id             = var.ws_os_image
  ws_collab_ssh_public_key       = var.ws_ssh_public_key
  ws_collab_cores                = local.paid_tier.websocket.cores
  ws_collab_memory_gb            = local.paid_tier.websocket.memory_gb
  ws_collab_disk_size_gb         = local.paid_tier.websocket.disk_size
  ws_collab_enable_nat           = local.global_config.defaults.assign_public_ip
  ws_collab_environment          = "paid"

  teleport_auth_server = module.bastion.bastion_private_ip
  teleport_auth_token  = random_password.teleport_token.result
  teleport_deb_base64  = local.teleport_deb_base64
}