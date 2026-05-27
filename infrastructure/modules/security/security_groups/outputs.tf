# # ============================================
# # EXISTING SECURITY GROUP OUTPUTS
# # ============================================

# output "ml_service_security_group_id" {
#   description = "ID of the ML service security group"
#   value       = yandex_vpc_security_group.ml_service_security_group.id
# }

# output "websocket_collab_security_group_id" {
#   description = "ID of the WebSocket collaboration security group"
#   value       = yandex_vpc_security_group.websocket_collab_security_group.id
# }

# output "backend_server_security_group_id" {
#   description = "ID of the backend server security group"
#   value       = yandex_vpc_security_group.backend_server_security_group.id
# }

# output "frontend_security_group_id" {
#   description = "ID of the frontend server security group"
#   value       = yandex_vpc_security_group.frontend_security_group.id
# }

# output "bastion_security_group_id" {
#   description = "ID of the bastion host security group"
#   value       = yandex_vpc_security_group.bastion_security_group.id
# }

# output "ydb_security_group_id" {
#   description = "ID of the YDB/PostgreSQL database security group"
#   value       = yandex_vpc_security_group.ydb_security_group.id
# }

# # ============================================
# # NEW CONSUL SECURITY GROUP OUTPUTS
# # ============================================

# output "consul_server_security_group_id" {
#   description = "ID of the Consul server security group"
#   value       = yandex_vpc_security_group.consul_server_security_group.id
# }

# output "consul_security_group_ids" {
#   description = "All Consul-related security group IDs"
#   value = {
#     server  = yandex_vpc_security_group.consul_server_security_group.id
#     sidecar = {
#       ml        = yandex_vpc_security_group.ml_service_security_group.id
#       backend   = yandex_vpc_security_group.backend_server_security_group.id
#       frontend  = yandex_vpc_security_group.frontend_security_group.id
#       websocket = yandex_vpc_security_group.websocket_collab_security_group.id
#     }
#   }
# }

# # ============================================
# # NETWORK CONFIGURATION SUMMARY
# # ============================================

# output "service_mesh_ports" {
#   description = "Consul service mesh ports used in security groups"
#   value = {
#     consul_rpc        = 8300
#     consul_serf_lan   = 8301
#     consul_serf_wan   = 8302
#     consul_http_api   = 8500
#     consul_https_ui   = 8501
#     consul_grpc       = 8502
#     consul_dns        = 8600
#     sidecar_proxy     = 15001
#     envoy_metrics     = 9090
#   }
# }

# output "security_group_summary" {
#   description = "Summary of all security groups and their purposes"
#   value = {
#     ml_service = {
#       id          = yandex_vpc_security_group.ml_service_security_group.id
#       description = "ML service with Consul sidecar"
#       services    = ["ml-api", "model-serving", "consul-sidecar"]
#     }
#     websocket = {
#       id          = yandex_vpc_security_group.websocket_collab_security_group.id
#       description = "WebSocket collaboration with Consul mesh"
#       services    = ["websocket", "consul-sidecar"]
#     }
#     backend = {
#       id          = yandex_vpc_security_group.backend_server_security_group.id
#       description = "Backend API with Consul sidecar"
#       services    = ["backend-api", "consul-sidecar"]
#     }
#     frontend = {
#       id          = yandex_vpc_security_group.frontend_security_group.id
#       description = "Frontend with Consul sidecar"
#       services    = ["frontend", "consul-sidecar"]
#     }
#     bastion = {
#       id          = yandex_vpc_security_group.bastion_security_group.id
#       description = "Bastion host for admin access"
#       services    = ["ssh", "consul-ui-proxy"]
#     }
#     database = {
#       id          = yandex_vpc_security_group.ydb_security_group.id
#       description = "PostgreSQL database cluster"
#       services    = ["postgresql"]
#     }
#     consul_server = {
#       id          = yandex_vpc_security_group.consul_server_security_group.id
#       description = "Consul server cluster"
#       services    = ["consul-server", "consul-dns", "consul-api"]
#     }
#   }
# }