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
  global_config = jsondecode(file("${var.config_dir}/defaults/global.json"))
  free_tier     = jsondecode(file("${var.config_dir}/free_tier.json"))
  paid_tier     = jsondecode(file("${var.config_dir}/paid_tier.json"))
  environment   = local.global_config.environment
}

# ============================================================
# RANDOM PASSWORDS
# ============================================================
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
    { zone = "ru-central1-c", cidr = "10.0.22.0/24" },
  ]
  net_public_subnets = [
    { zone = "ru-central1-a", cidr = "10.0.10.0/24" }
  ]
}

# ============================================================
# STATIC IPS
# ============================================================
module "alb_static_ip" {
  source      = "../../modules/core/static_ip"
  name        = "${local.environment}-alb-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for ALB"
  labels      = { environment = local.environment, service = "alb" }
  deletion_protection = true
}

module "bastion_static_ip" {
  source      = "../../modules/core/static_ip"
  name        = "${local.environment}-bastion-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for bastion"
  labels      = { environment = local.environment, service = "bastion" }
  deletion_protection = true
}

module "observability_static_ip" {
  source      = "../../modules/core/static_ip"
  name        = "${local.environment}-observability-ip"
  folder_id   = var.yandex_cloud_folder_id
  zone_id     = "ru-central1-a"
  description = "Static IP for observability server"
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
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "observability" }
}

resource "yandex_vpc_security_group" "nomad_server_security_group" {
  name        = var.nomad_server_sg_name
  description = "Nomad server cluster security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "Nomad RPC"
    port           = 4647
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Nomad Serf TCP"
    port           = 4648
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "UDP"
    description    = "Nomad Serf UDP"
    port           = 4648
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Nomad HTTP API"
    port           = 4646
    v4_cidr_blocks = var.admin_ip_cidr
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "nomad" }
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
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
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
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
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
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
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
    description    = "WebSocket from ALB"
    port           = 8080
    v4_cidr_blocks = ["10.0.10.0/24"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
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
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "bastion" }
}

resource "yandex_vpc_security_group" "security_tools_security_group" {
  name        = var.security_tools_sg_name
  description = "Security tools (SIEM/SOAR) security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "Web UIs from admin"
    from_port      = 8080
    to_port        = 9090
    v4_cidr_blocks = var.admin_ip_cidr
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "security-tools" }
}

resource "yandex_vpc_security_group" "consul_server_security_group" {
  name        = var.consul_server_sg_name
  description = "Consul server cluster security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  ingress {
    protocol       = "TCP"
    description    = "Consul RPC"
    port           = 8300
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Consul Serf LAN TCP"
    port           = 8301
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "UDP"
    description    = "Consul Serf LAN UDP"
    port           = 8301
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Consul Serf WAN TCP"
    port           = 8302
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "UDP"
    description    = "Consul Serf WAN UDP"
    port           = 8302
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Consul HTTP API"
    port           = 8500
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "Consul gRPC"
    port           = 8502
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "UDP"
    description    = "Consul DNS"
    port           = 8600
    v4_cidr_blocks = ["10.0.0.0/8"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from bastion"
    port           = 22
    security_group_id = yandex_vpc_security_group.bastion_security_group.id
  }
  egress {
    protocol       = "ANY"
    description    = "All outbound"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  labels = { environment = local.environment, service = "consul" }
}

resource "yandex_vpc_security_group" "github_runner_security_group" {
  name        = var.github_runner_sg_name
  description = "GitHub runner security group"
  network_id  = module.vpc.vpc_id
  folder_id   = var.yandex_cloud_folder_id

  egress {
    protocol       = "ANY"
    description    = "All outbound (needed to reach GitHub API, download dependencies)"
    v4_cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    protocol       = "TCP"
    description    = "SSH from admin"
    port           = 22
    v4_cidr_blocks = var.admin_ip_cidr
  }
  labels = { environment = local.environment, service = "github-runner" }
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
# GENERIC COMPUTE INSTANCES
# ============================================================

module "bastion" {
  source = "../../modules/security/bastion"

  folder_id           = var.yandex_cloud_folder_id
  environment         = local.environment
  subnet_id           = module.vpc.public_subnet_ids["ru-central1-a"]
  instance_name       = "bastion-${local.environment}"
  zone                = "ru-central1-a"
  assign_public_ip    = true
  static_ip_address   = module.bastion_static_ip.ip_address
  cores               = local.global_config.bastion.cores
  ram_gb              = local.global_config.bastion.memory_gb
  boot_disk_size_gb   = local.global_config.bastion.disk_size
  ssh_user            = "ubuntu"
  ssh_public_key      = var.bastion_ssh_public_key
  security_group_ids  = [yandex_vpc_security_group.bastion_security_group.id]
}

module "observability" {
  source = "../../modules/observability"

  folder_id           = var.yandex_cloud_folder_id
  environment         = local.environment
  subnet_id           = module.vpc.public_subnet_ids["ru-central1-a"]
  instance_name       = "observability-${local.environment}"
  zone                = "ru-central1-a"
  assign_public_ip    = true
  static_ip_address   = module.observability_static_ip.ip_address
  cores               = local.global_config.observability.cores
  ram_gb              = local.global_config.observability.memory_gb
  boot_disk_size_gb   = local.global_config.observability.boot_disk_size
  data_disk_enabled   = local.global_config.observability.data_disk_enabled
  data_disk_size_gb   = local.global_config.observability.data_disk_size
  ssh_user            = "ubuntu"
  ssh_public_key      = var.observability_ssh_public_key
  security_group_ids  = [yandex_vpc_security_group.observability_security_group.id]
}

module "security_tools" {
  source = "../../modules/security/monitoring"

  folder_id           = var.yandex_cloud_folder_id
  environment         = local.environment
  instance_name       = "security-tools-${local.environment}"
  zone                = "ru-central1-a"
  subnet_id           = module.vpc.private_subnet_ids["ru-central1-a"]
  security_group_ids  = [yandex_vpc_security_group.security_tools_security_group.id]
  assign_public_ip    = false
  static_ip_address   = null
  cores               = var.security_tools_cores
  ram_gb              = var.security_tools_ram_gb
  core_fraction       = 100
  platform_id         = "standard-v3"
  preemptible         = false
  boot_disk_size_gb   = var.security_tools_boot_disk_size
  boot_disk_type      = "network-ssd"
  data_disk_enabled   = var.security_tools_data_disk_enabled
  data_disk_size_gb   = var.security_tools_data_disk_size
  data_disk_type      = "network-ssd"
  os_image_family     = "ubuntu-2204-lts"
  ssh_user            = "ubuntu"
  ssh_public_key      = var.security_tools_ssh_public_key
  user_data           = ""
}

module "github_runner" {
  source = "../../modules/services/ci_cd_runner"

  folder_id           = var.yandex_cloud_folder_id
  environment         = local.environment
  subnet_id           = module.vpc.private_subnet_ids["ru-central1-a"]
  security_group_ids  = [yandex_vpc_security_group.github_runner_security_group.id]
  zone                = "ru-central1-a"
  assign_public_ip    = true
  static_ip_address   = null
  cores               = var.github_runner_cores
  ram_gb              = var.github_runner_ram_gb
  boot_disk_size_gb   = var.github_runner_boot_disk_size
  data_disk_enabled   = var.github_runner_data_disk_enabled
  data_disk_size_gb   = var.github_runner_data_disk_size
  ssh_user            = "ubuntu"
  ssh_public_key      = var.github_runner_ssh_public_key
  user_data           = var.github_runner_user_data
}

# ============================================================
# FREE TIER SERVICES
# ============================================================
module "backend_free" {
  source = "../../modules/services/backend_service"

  instance_count = local.free_tier.backend.instance_count
  instance_name_prefix = "backend-free-${local.environment}"
  folder_id      = var.yandex_cloud_folder_id
  environment    = "free"
  subnet_id      = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  zones          = [local.free_tier.zone]
  cores          = local.free_tier.backend.cores
  ram_bytes        = local.free_tier.backend.memory_gb * 1024 * 1024 * 1024
  boot_disk_size_gb = local.free_tier.backend.disk_size
  ssh_user       = "ubuntu"
  ssh_public_key = var.backend_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.backend_server_security_group.id]
  create_target_group = true
  assign_public_ip = false
}

module "frontend_free" {
  source = "../../modules/services/frontend_service"

  instance_name  = "frontend-free-${local.environment}"
  folder_id      = var.yandex_cloud_folder_id
  environment    = "free"
  subnet_id      = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  zone           = local.free_tier.zone
  cores          = local.free_tier.frontend.cores
  ram_bytes        = local.free_tier.frontend.memory_gb * 1024 * 1024 * 1024
  boot_disk_size_gb = local.free_tier.frontend.disk_size
  ssh_user       = "ubuntu"
  ssh_public_key = var.frontend_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.frontend_security_group.id]
  assign_public_ip = false
  user_data      = ""
}

module "ml_free" {
  source = "../../modules/services/ml_service"

  instance_name    = "ml-free-${local.environment}"
  folder_id        = var.yandex_cloud_folder_id
  environment      = "free"
  subnet_id        = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  zone             = local.free_tier.zone
  cores            = local.free_tier.ml.cores
  ram_gb           = local.free_tier.ml.memory_gb
  gpus             = local.free_tier.ml.gpus
  boot_disk_size_gb = local.free_tier.ml.disk_size
  data_disk_enabled = true
  data_disk_size_gb = 200
  ssh_user         = "ubuntu"
  ssh_public_key   = var.ml_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.ml_service_security_group.id]
  assign_public_ip = false
  platform_id      = local.free_tier.ml.platform
  user_data        = ""
}

module "websocket_free" {
  source = "../../modules/services/websocket_collab_service"

  instance_name    = "ws-free-${local.environment}"
  folder_id        = var.yandex_cloud_folder_id
  environment      = "free"
  subnet_id        = module.vpc.private_subnet_ids[local.free_tier.subnet_key]
  zone             = local.free_tier.zone
  cores            = local.free_tier.websocket.cores
  ram_gb           = local.free_tier.websocket.memory_gb
  boot_disk_size_gb = local.free_tier.websocket.disk_size
  app_port         = 8080
  ssh_user         = "ubuntu"
  ssh_public_key   = var.ws_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.websocket_collab_security_group.id]
  assign_public_ip = false
  create_alb_resources = false
  user_data        = ""
}

# ============================================================
# PAID TIER SERVICES
# ============================================================
module "backend_paid" {
  source = "../../modules/services/backend_service"

  instance_count = local.paid_tier.backend.instance_count
  instance_name_prefix = "backend-paid-${local.environment}"
  folder_id      = var.yandex_cloud_folder_id
  environment    = "paid"
  subnet_id      = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  zones          = [local.paid_tier.zone]
  cores          = local.paid_tier.backend.cores
  ram_bytes        = local.paid_tier.backend.memory_gb * 1024 * 1024 * 1024
  boot_disk_size_gb = local.paid_tier.backend.disk_size
  ssh_user       = "ubuntu"
  ssh_public_key = var.backend_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.backend_server_security_group.id]
  create_target_group = true
  assign_public_ip = false
}

module "frontend_paid" {
  source = "../../modules/services/frontend_service"

  instance_name  = "frontend-paid-${local.environment}"
  folder_id      = var.yandex_cloud_folder_id
  environment    = "paid"
  subnet_id      = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  zone           = local.paid_tier.zone
  cores          = local.paid_tier.frontend.cores
  ram_bytes        = local.paid_tier.frontend.memory_gb * 1024 * 1024 * 1024
  boot_disk_size_gb = local.paid_tier.frontend.disk_size
  ssh_user       = "ubuntu"
  ssh_public_key = var.frontend_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.frontend_security_group.id]
  assign_public_ip = false
  user_data      = ""
}

module "ml_paid" {
  source = "../../modules/services/ml_service"

  instance_name    = "ml-paid-${local.environment}"
  folder_id        = var.yandex_cloud_folder_id
  environment      = "paid"
  subnet_id        = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  zone             = local.paid_tier.zone
  cores            = local.paid_tier.ml.cores
  ram_gb           = local.paid_tier.ml.memory_gb
  gpus             = local.paid_tier.ml.gpus
  boot_disk_size_gb = local.paid_tier.ml.disk_size
  data_disk_enabled = true
  data_disk_size_gb = 200
  ssh_user         = "ubuntu"
  ssh_public_key   = var.ml_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.ml_service_security_group.id]
  assign_public_ip = false
  platform_id      = local.paid_tier.ml.platform
  user_data        = ""
}

module "websocket_paid" {
  source = "../../modules/services/websocket_collab_service"

  instance_name    = "ws-paid-${local.environment}"
  folder_id        = var.yandex_cloud_folder_id
  environment      = "paid"
  subnet_id        = module.vpc.private_subnet_ids[local.paid_tier.subnet_key]
  zone             = local.paid_tier.zone
  cores            = local.paid_tier.websocket.cores
  ram_gb           = local.paid_tier.websocket.memory_gb
  boot_disk_size_gb = local.paid_tier.websocket.disk_size
  app_port         = 8080
  ssh_user         = "ubuntu"
  ssh_public_key   = var.ws_ssh_public_key
  security_group_ids = [yandex_vpc_security_group.websocket_collab_security_group.id]
  assign_public_ip = false
  create_alb_resources = false
  user_data        = ""
}

# ============================================================
# CONSUL SERVER CLUSTER
# ============================================================
module "consul" {
  source = "../../modules/core/consul"

  folder_id           = var.yandex_cloud_folder_id
  environment         = local.environment
  private_subnet_ids  = module.vpc.private_subnet_ids
  security_group_id   = yandex_vpc_security_group.consul_server_security_group.id

  instance_count      = var.consul_servers_count
  instance_cores      = var.consul_server_cores
  instance_ram_gb     = var.consul_server_memory_gb
  instance_disk_size_gb = var.consul_server_disk_size
  instance_disk_type    = var.consul_server_disk_type
  instance_platform_id  = var.consul_server_platform_id
  preemptible         = false

  os_family           = "ubuntu-2204-lts"
  ssh_user            = "ubuntu"
  ssh_public_key      = var.backend_ssh_public_key

  create_target_group = var.consul_create_load_balancer
}

# ============================================================
# NOMAD FREE TIER CLUSTER
# ============================================================
module "nomad_free" {
  source = "../../modules/core/nomad"

  folder_id           = var.yandex_cloud_folder_id
  environment         = "free"
  private_subnet_ids  = { for k, v in module.vpc.private_subnet_ids : k => v if k == local.free_tier.subnet_key }
  security_group_id   = yandex_vpc_security_group.nomad_server_security_group.id
  instance_count      = var.nomad_free_servers_count
  instance_cores      = var.nomad_free_server_cores
  instance_ram_gb     = var.nomad_free_server_memory_gb
  instance_disk_size_gb = var.nomad_free_server_disk_size
  instance_disk_type    = var.nomad_free_server_disk_type
  instance_platform_id  = "standard-v3"
  preemptible         = var.nomad_free_preemptible

  os_family           = "ubuntu-2204-lts"
  ssh_user            = "ubuntu"
  ssh_public_key      = var.backend_ssh_public_key

  create_target_group = false
}

# ============================================================
# NOMAD PAID TIER CLUSTER
# ============================================================
module "nomad_paid" {
  source = "../../modules/core/nomad"

  folder_id           = var.yandex_cloud_folder_id
  environment         = "paid"
  private_subnet_ids  = { for k, v in module.vpc.private_subnet_ids : k => v if k == local.paid_tier.subnet_key }
  security_group_id   = yandex_vpc_security_group.nomad_server_security_group.id

  instance_count      = var.nomad_paid_servers_count
  instance_cores      = var.nomad_paid_server_cores
  instance_ram_gb     = var.nomad_paid_server_memory_gb
  instance_disk_size_gb = var.nomad_paid_server_disk_size
  instance_disk_type    = var.nomad_paid_server_disk_type
  instance_platform_id  = "standard-v3"
  preemptible         = var.nomad_paid_preemptible

  os_family           = "ubuntu-2204-lts"
  ssh_user            = "ubuntu"
  ssh_public_key      = var.backend_ssh_public_key

  create_target_group = false
}

# ============================================================
# S3 BUCKET (for state and backups)
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