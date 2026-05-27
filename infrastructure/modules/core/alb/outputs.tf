output "load_balancer_id" {
  description = "ALB instance ID"
  value       = yandex_alb_load_balancer.main.id
}

output "load_balancer_name" {
  description = "ALB instance name"
  value       = yandex_alb_load_balancer.main.name
}

output "load_balancer_ip" {
  description = "ALB external IP address"
  value       = yandex_alb_load_balancer.main.listener[0].endpoint[0].address[0].external_ipv4_address[0].address
}

output "http_router_id" {
  description = "HTTP router ID"
  value       = yandex_alb_http_router.main.id
}

output "virtual_host_name" {
  description = "Virtual host name"
  value       = yandex_alb_virtual_host.main.name
}

output "listener_ports" {
  description = "ALB listener ports"
  value = {
    http = 80
  }
}

output "alb_endpoint" {
  description = "ALB endpoint URL"
  value       = "http://${yandex_alb_load_balancer.main.listener[0].endpoint[0].address[0].external_ipv4_address[0].address}"
}