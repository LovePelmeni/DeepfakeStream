# # ============================================
# # VPC AND NETWORKING VARIABLES
# # ============================================

# variable "vpc_id" {
#   description = "ID of the VPC network where security groups will be created"
#   type        = string
# }

# variable "public_v4_cidr_blocks" {
#   description = "List of IPv4 CIDR blocks for public subnets (e.g., from VPC module)"
#   type        = list(string)
# }

# variable "private_v4_cidr_blocks" {
#   description = "List of IPv4 CIDR blocks for private subnets"
#   type        = list(string)
# }

# variable "bastion_metadata_ip_cidr" {
#   description = "CIDR block(s) allowed for bastion metadata access (port 22 egress)"
#   type        = list(string)
# }

# variable "ml_metadata_ip_cidr" {
#   description = "CIDR block(s) allowed for metadata access (port 80 egress)"
#   type        = list(string)
# }

# variable "ml_dns_ip_cidr" {
#   description = "CIDR block(s) allowed for DNS (port 53 UDP egress)"
#   type        = list(string)
# }

# variable "bastion_admin_ip_cidr" {
#   description = "CIDR block(s) allowed for SSH access to bastion host"
#   type        = list(string)
# }

# variable "metrics_cidr_blocks" {
#   description = "CIDR blocks for Prometheus metrics scraping"
#   type        = list(string)
#   default     = []
# }

# # ============================================
# # CONSUL NETWORKING VARIABLES
# # ============================================

# variable "consul_server_cidrs" {
#   description = "CIDR blocks for Consul server cluster"
#   type        = list(string)
#   default     = []  # Will be set by VPC module
# }

# variable "environment" {
#   description = "Environment name (e.g., dev, staging, prod)"
#   type        = string
#   default     = "prod"
# }

# # ============================================
# # SECURITY GROUP NAMES
# # ============================================

# variable "ml_service_sg_name" {
#   description = "Name of the ML service security group"
#   type        = string
# }

# variable "ws_collab_sg_name" {
#   description = "Name of the WebSocket collaboration security group"
#   type        = string
# }

# variable "backend_server_sg_name" {
#   description = "Name of the backend server security group"
#   type        = string
# }

# variable "frontend_server_sg_name" {
#   description = "Name of the frontend server security group"
#   type        = string
# }

# variable "bastion_sg_name" {
#   description = "Name of the bastion host security group"
#   type        = string
# }

# variable "pg_ydb_sg_name" {
#   description = "Name of the PostgreSQL/YDB database security group"
#   type        = string
# }

# variable "consul_server_sg_name" {
#   description = "Name of the Consul server security group"
#   type        = string
#   default     = "consul-server-sg"
# }

# # ============================================
# # SERVICE TAGS (for Consul service discovery)
# # ============================================

# variable "consul_enabled" {
#   description = "Enable Consul service mesh integration"
#   type        = bool
#   default     = true
# }

# variable "consul_datacenter" {
#   description = "Consul datacenter name"
#   type        = string
#   default     = "yc-ru-central1"
# }

# # ============================================
# # PORT NUMBERS (to avoid hardcoding)
# # ============================================

# variable "port_http" {
#   description = "HTTP port"
#   type        = number
#   default     = 80
# }

# variable "port_https" {
#   description = "HTTPS port"
#   type        = number
#   default     = 443
# }

# variable "port_ssh" {
#   description = "SSH port"
#   type        = number
#   default     = 22
# }

# variable "port_dns_udp" {
#   description = "DNS UDP port"
#   type        = number
#   default     = 53
# }

# variable "port_metadata" {
#   description = "Metadata service port"
#   type        = number
#   default     = 80
# }

# variable "port_postgres" {
#   description = "PostgreSQL port"
#   type        = number
#   default     = 5432
# }

# variable "port_postgres_ydb" {
#   description = "YDB/PostgreSQL port (managed)"
#   type        = number
#   default     = 6432
# }

# variable "port_grafana" {
#   description = "Grafana UI port"
#   type        = number
#   default     = 3000
# }

# variable "port_prometheus" {
#   description = "Prometheus UI port"
#   type        = number
#   default     = 9090
# }

# variable "port_mlflow" {
#   description = "MLflow UI port"
#   type        = number
#   default     = 5000
# }

# variable "port_teleport_auth" {
#   description = "Teleport auth port"
#   type        = number
#   default     = 3025
# }

# variable "port_teleport_proxy" {
#   description = "Teleport SSH proxy port"
#   type        = number
#   default     = 3022
# }

# # Consul ports
# variable "port_consul_rpc" {
#   description = "Consul RPC port"
#   type        = number
#   default     = 8300
# }

# variable "port_consul_serf_lan_tcp" {
#   description = "Consul Serf LAN TCP"
#   type        = number
#   default     = 8301
# }

# variable "port_consul_serf_lan_udp" {
#   description = "Consul Serf LAN UDP"
#   type        = number
#   default     = 8301
# }

# variable "port_consul_serf_wan_tcp" {
#   description = "Consul Serf WAN TCP"
#   type        = number
#   default     = 8302
# }

# variable "port_consul_serf_wan_udp" {
#   description = "Consul Serf WAN UDP"
#   type        = number
#   default     = 8302
# }

# variable "port_consul_http_api" {
#   description = "Consul HTTP API"
#   type        = number
#   default     = 8500
# }

# variable "port_consul_https_ui" {
#   description = "Consul HTTPS UI"
#   type        = number
#   default     = 8501
# }

# variable "port_consul_grpc" {
#   description = "Consul gRPC for sidecars"
#   type        = number
#   default     = 8502
# }

# variable "port_consul_dns_udp" {
#   description = "Consul DNS UDP"
#   type        = number
#   default     = 8600
# }

# variable "port_consul_sidecar_proxy" {
#   description = "Consul sidecar proxy port"
#   type        = number
#   default     = 15001
# }

# variable "port_app_health" {
#   description = "Application health check port"
#   type        = number
#   default     = 8080
# }

# variable "port_app_metrics" {
#   description = "Application metrics port"
#   type        = number
#   default     = 9090
# }