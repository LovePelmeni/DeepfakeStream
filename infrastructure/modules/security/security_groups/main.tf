# # ============================================
# # EXISTING SECURITY GROUPS (with Consul ports added)
# # ============================================

# # ML service security group
# resource "yandex_vpc_security_group" "ml_service_security_group" {
#   name        = var.ml_service_sg_name
#   description = "ML service security group with Consul sidecar support"
#   network_id  = var.vpc_id

#   # Inbound: TLS (port 443)
#   ingress {
#     protocol       = "TCP"
#     from_port      = 443
#     to_port        = 443
#     v4_cidr_blocks = ["0.0.0.0/0"]
#   }

#   # Inbound: Health checks (port 8080) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8080
#     to_port        = 8080
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "Health checks"
#   }

#   # Inbound: Consul sidecar proxy (port 15001)
#   ingress {
#     protocol       = "TCP"
#     from_port      = 15001
#     to_port        = 15001
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul sidecar proxy traffic"
#   }

#   # Inbound: Consul gRPC (port 8502) - for sidecar communication
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8502
#     to_port        = 8502
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul gRPC for service mesh"
#   }

#   # Inbound: Application metrics (optional)
#   ingress {
#     protocol       = "TCP"
#     from_port      = 9090
#     to_port        = 9090
#     v4_cidr_blocks = var.metrics_cidr_blocks
#     description    = "Prometheus metrics scraping"
#   }

#   # Outbound: SSH to bastion (or metadata)
#   egress {
#     protocol       = "TCP"
#     from_port      = 22
#     to_port        = 22
#     v4_cidr_blocks = var.bastion_metadata_ip_cidr
#     description    = "SSH to bastion"
#   }

#   # Outbound: Metadata (port 80)
#   egress {
#     protocol       = "TCP"
#     from_port      = 80
#     to_port        = 80
#     v4_cidr_blocks = var.ml_metadata_ip_cidr
#     description    = "Metadata"
#   }

#   # Outbound: DNS (port 53 UDP)
#   egress {
#     protocol       = "UDP"
#     from_port      = 53
#     to_port        = 53
#     v4_cidr_blocks = var.ml_dns_ip_cidr
#     description    = "DNS"
#   }

#   # Outbound: Consul server communication
#   egress {
#     protocol       = "TCP"
#     from_port      = 8500
#     to_port        = 8502
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul server API and gRPC"
#   }

#   tags = {
#     Name        = var.ml_service_sg_name
#     Environment = var.environment
#     Service     = "ml"
#     ConsulMesh  = "enabled"
#   }
# }

# # WebSocket collaboration security group
# resource "yandex_vpc_security_group" "websocket_collab_security_group" {
#   name        = var.ws_collab_sg_name
#   description = "Websocket collaboration server security group with Consul mesh"
#   network_id  = var.vpc_id

#   # Inbound: WebSocket TLS (443) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 443
#     to_port        = 443
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "WebSocket TLS from Load Balancer"
#   }

#   # Inbound: WebSocket (80) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 80
#     to_port        = 80
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "WebSocket from Load Balancer (fallback)"
#   }

#   # Inbound: Health checks (8080) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8080
#     to_port        = 8080
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "Health checks from ALB"
#   }

#   # Inbound: Consul sidecar proxy
#   ingress {
#     protocol       = "TCP"
#     from_port      = 15001
#     to_port        = 15001
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul sidecar proxy"
#   }

#   # Inbound: Consul gRPC
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8502
#     to_port        = 8502
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul service mesh gRPC"
#   }

#   # Outbound: VPC internal traffic to private subnets
#   egress {
#     protocol       = "ANY"
#     from_port      = 0
#     to_port        = 65535
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "VPC internal traffic"
#   }

#   # Outbound: Consul servers
#   egress {
#     protocol       = "TCP"
#     from_port      = 8500
#     to_port        = 8502
#     v4_cidr_blocks = var.consul_server_cidrs
#     description    = "Consul server communication"
#   }

#   tags = {
#     Name        = var.ws_collab_sg_name
#     Environment = var.environment
#     Service     = "websocket"
#     ConsulMesh  = "enabled"
#   }
# }

# # Backend server security group
# resource "yandex_vpc_security_group" "backend_server_security_group" {
#   name        = var.backend_server_sg_name
#   description = "Backend server security group with Consul sidecar"
#   network_id  = var.vpc_id

#   # Inbound: Health checks (8080) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8080
#     to_port        = 8080
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "Health checks from ALB"
#   }

#   # Inbound: Consul sidecar proxy
#   ingress {
#     protocol       = "TCP"
#     from_port      = 15001
#     to_port        = 15001
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul sidecar proxy traffic"
#   }

#   # Inbound: Consul gRPC
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8502
#     to_port        = 8502
#     v4_cidr_blocks = var.consul_server_cidrs
#     description    = "Consul service mesh registration"
#   }

#   # Inbound: Application metrics
#   ingress {
#     protocol       = "TCP"
#     from_port      = 9090
#     to_port        = 9090
#     v4_cidr_blocks = var.metrics_cidr_blocks
#     description    = "Prometheus metrics"
#   }

#   # Outbound: External APIs and VPC services
#   egress {
#     protocol       = "ANY"
#     from_port      = 0
#     to_port        = 65535
#     v4_cidr_blocks = ["0.0.0.0/0"]
#     description    = "External APIs + VPC services (S3, endpoints)"
#   }

#   tags = {
#     Name        = var.backend_server_sg_name
#     Environment = var.environment
#     Service     = "backend"
#     ConsulMesh  = "enabled"
#   }
# }

# # Frontend security group
# resource "yandex_vpc_security_group" "frontend_security_group" {
#   name        = var.frontend_server_sg_name
#   description = "Frontend server security group with Consul sidecar"
#   network_id  = var.vpc_id

#   # Inbound: Health checks (8080) from public subnets
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8080
#     to_port        = 8080
#     v4_cidr_blocks = var.public_v4_cidr_blocks
#     description    = "Health checks from ALB"
#   }

#   # Inbound: Consul sidecar proxy
#   ingress {
#     protocol       = "TCP"
#     from_port      = 15001
#     to_port        = 15001
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "Consul sidecar proxy"
#   }

#   # Inbound: Consul gRPC
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8502
#     to_port        = 8502
#     v4_cidr_blocks = var.consul_server_cidrs
#     description    = "Consul mesh registration"
#   }

#   # Outbound: External APIs and VPC services
#   egress {
#     protocol       = "ANY"
#     from_port      = 0
#     to_port        = 65535
#     v4_cidr_blocks = ["0.0.0.0/0"]
#     description    = "External APIs + VPC services (S3, endpoints)"
#   }

#   tags = {
#     Name        = var.frontend_server_sg_name
#     Environment = var.environment
#     Service     = "frontend"
#     ConsulMesh  = "enabled"
#   }
# }

# # Bastion host security group
# resource "yandex_vpc_security_group" "bastion_security_group" {
#   name        = var.bastion_sg_name
#   description = "Bastion host security group - admin SSH access only"
#   network_id  = var.vpc_id

#   # Inbound: SSH from admin IP only
#   ingress {
#     protocol       = "TCP"
#     from_port      = 22
#     to_port        = 22
#     v4_cidr_blocks = var.bastion_admin_ip_cidr
#     description    = "Admin SSH access"
#   }

#   # Inbound: Consul UI access (optional, via SSH tunnel)
#   ingress {
#     protocol       = "TCP"
#     from_port      = 8500
#     to_port        = 8500
#     v4_cidr_blocks = var.bastion_admin_ip_cidr
#     description    = "Consul UI access via bastion"
#   }

#   # Outbound: VPC internal traffic to private subnets
#   egress {
#     protocol       = "ANY"
#     from_port      = 0
#     to_port        = 65535
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#     description    = "VPC internal"
#   }

#   tags = {
#     Name        = var.bastion_sg_name
#     Environment = var.environment
#     Service     = "bastion"
#   }
# }

# # YDB / PostgreSQL database security group
# resource "yandex_vpc_security_group" "ydb_security_group" {
#   name        = var.pg_ydb_sg_name
#   description = "PostgreSQL/YDB database security group with Consul mesh"
#   network_id  = var.vpc_id

#   # Allow DB access from backend server (by security group ID)
#   ingress {
#     protocol          = "TCP"
#     description       = "Postgres from backend VMs"
#     port              = 6432
#     security_group_id = yandex_vpc_security_group.backend_server_security_group.id
#   }

#   # Allow DB access from ML service
#   ingress {
#     protocol          = "TCP"
#     description       = "Postgres from ML service"
#     port              = 6432
#     security_group_id = yandex_vpc_security_group.ml_service_security_group.id
#   }

#   # Optional: Consul service mesh can also access database
#   ingress {
#     protocol          = "TCP"
#     description       = "Postgres from Consul mesh services"
#     port              = 6432
#     v4_cidr_blocks    = var.consul_server_cidrs
#   }

#   # Outbound: all
#   egress {
#     protocol       = "ANY"
#     description    = "All outbound"
#     v4_cidr_blocks = ["0.0.0.0/0"]
#   }

#   tags = {
#     Name        = var.pg_ydb_sg_name
#     Environment = var.environment
#     Service     = "database"
#   }
# }

# # ============================================
# # NEW CONSUL SERVER SECURITY GROUP
# # ============================================

# resource "yandex_vpc_security_group" "consul_server_security_group" {
#   name        = var.consul_server_sg_name
#   description = "Consul server cluster security group"
#   network_id  = var.vpc_id

#   # Consul RPC (server to server)
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul RPC"
#     port           = 8300
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # Consul Serf LAN (gossip protocol)
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul Serf LAN TCP"
#     port           = 8301
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }
  
#   ingress {
#     protocol       = "UDP"
#     description    = "Consul Serf LAN UDP"
#     port           = 8301
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # Consul Serf WAN
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul Serf WAN TCP"
#     port           = 8302
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }
  
#   ingress {
#     protocol       = "UDP"
#     description    = "Consul Serf WAN UDP"
#     port           = 8302
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # HTTP API
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul HTTP API"
#     port           = 8500
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # HTTPS UI (optional, from bastion/admin)
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul HTTPS UI"
#     port           = 8501
#     v4_cidr_blocks = var.bastion_admin_ip_cidr
#   }

#   # gRPC for sidecars
#   ingress {
#     protocol       = "TCP"
#     description    = "Consul gRPC for sidecars"
#     port           = 8502
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # DNS interface
#   ingress {
#     protocol       = "UDP"
#     description    = "Consul DNS"
#     port           = 8600
#     v4_cidr_blocks = var.private_v4_cidr_blocks
#   }

#   # Internal traffic between Consul servers
#   ingress {
#     protocol       = "TCP"
#     description    = "Internal Consul traffic"
#     from_port      = 8300
#     to_port        = 8502
#     v4_cidr_blocks = var.consul_server_cidrs
#   }

#   # SSH from bastion
#   ingress {
#     protocol          = "TCP"
#     description       = "SSH from bastion"
#     port              = 22
#     security_group_id = yandex_vpc_security_group.bastion_security_group.id
#   }

#   # All outbound
#   egress {
#     protocol       = "ANY"
#     description    = "All outbound traffic"
#     v4_cidr_blocks = ["0.0.0.0/0"]
#   }

#   tags = {
#     Name        = var.consul_server_sg_name
#     Environment = var.environment
#     Service     = "consul"
#     Component   = "server"
#   }
# }