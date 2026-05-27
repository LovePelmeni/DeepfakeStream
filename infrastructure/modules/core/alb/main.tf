# Application Load Balancer with listeners defined inside
resource "yandex_alb_load_balancer" "main" {
  name        = "${var.name_prefix}-alb"
  folder_id   = var.folder_id
  network_id  = var.network_id

  allocation_policy {
    dynamic "location" {
      for_each = var.subnet_ids
      content {
        zone_id   = element(["ru-central1-a", "ru-central1-b", "ru-central1-d"], index(var.subnet_ids, location.value))
        subnet_id = location.value
      }
    }
  }

  security_group_ids = var.security_group_ids

  # HTTP Listener
  listener {
    name = "${var.name_prefix}-http-listener"
    endpoint {
      address {
        external_ipv4_address {
        }
      }
      ports = [80]
    }
    http {
      handler {
        http_router_id = yandex_alb_http_router.main.id
      }
    }
  }

  labels = var.labels
}

# HTTP Router
resource "yandex_alb_http_router" "main" {
  name      = "${var.name_prefix}-router"
  folder_id = var.folder_id
  labels    = var.labels
}

# Virtual Host with routing rules
resource "yandex_alb_virtual_host" "main" {
  name           = "${var.name_prefix}-vhost"
  http_router_id = yandex_alb_http_router.main.id

  # Routes for each tier
  dynamic "route" {
    for_each = var.tiers
    content {
      name = "${route.key}-route"
      
      http_route {
        http_route_action {
          backend_group_id = route.value.backend_group_id
          timeout          = "60s"
        }
      }
    }
  }

  # Default route
  dynamic "route" {
    for_each = var.default_backend_group_id != null ? [1] : []
    content {
      name = "default-route"
      
      http_route {
        http_route_action {
          backend_group_id = var.default_backend_group_id
          timeout          = "60s"
        }
      }
    }
  }
}