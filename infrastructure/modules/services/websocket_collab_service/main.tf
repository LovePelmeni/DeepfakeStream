# Look up OS image
data "yandex_compute_image" "os_image" {
  family = var.os_image_family
}

# Main compute instance
resource "yandex_compute_instance" "this" {
  name        = var.instance_name
  description = var.description
  folder_id   = var.folder_id
  zone        = var.zone
  platform_id = var.platform_id

  resources {
    cores         = var.cores
    core_fraction = var.core_fraction
    memory        = var.ram_gb * 1024 * 1024 * 1024
  }

  boot_disk {
    initialize_params {
      size     = var.boot_disk_size_gb
      type     = var.boot_disk_type
      image_id = var.image_id != "" ? var.image_id : data.yandex_compute_image.os_image.id
    }
  }

  network_interface {
    subnet_id          = var.subnet_id
    security_group_ids = var.security_group_ids
    nat                = var.assign_public_ip
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  metadata = merge(
    {
      ssh-keys = "${var.ssh_user}:${var.ssh_public_key}"
    },
    var.user_data != "" ? { user-data = var.user_data } : {},
    var.extra_metadata
  )

  lifecycle {
    create_before_destroy = true
  }

  labels = merge(
    {
      environment = var.environment
      purpose     = "websocket"
    },
    var.extra_labels
  )
}

# Optional: Target group for ALB integration
resource "yandex_alb_target_group" "this" {
  count = var.create_alb_resources ? 1 : 0

  name      = "${var.instance_name}-tg"
  folder_id = var.folder_id

  target {
    ip_address = yandex_compute_instance.this.network_interface[0].ip_address
    subnet_id  = var.subnet_id
  }

  labels = var.extra_labels
}

# Optional: Backend group for ALB integration
resource "yandex_alb_backend_group" "this" {
  count = var.create_alb_resources ? 1 : 0

  name      = "${var.instance_name}-bg"
  folder_id = var.folder_id

  http_backend {
    name = "websocket-backend"
    port = var.app_port
    target_group_ids = [yandex_alb_target_group.this[0].id]
    load_balancing_config {
      panic_threshold = 50
    }
    healthcheck {
      timeout          = "10s"
      interval         = "2s"
      healthcheck_port = var.app_port
      http_healthcheck {
        path = var.health_check_path
      }
    }
  }

  labels = var.extra_labels
}