# Resolve OS image ID
data "yandex_compute_image" "os_image" {
  family = var.os_image_family
}

# Main compute instance (bastion)
resource "yandex_compute_instance" "this" {
  name        = var.instance_name
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
      image_id = data.yandex_compute_image.os_image.id
    }
  }

  network_interface {
    subnet_id          = var.subnet_id
    security_group_ids = var.security_group_ids
    nat                = var.assign_public_ip
    ip_address         = var.static_ip_address != null ? var.static_ip_address : null
  }

  scheduling_policy {
    preemptible = var.preemptible
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${var.ssh_public_key}"
  }

  lifecycle {
    create_before_destroy = true
  }

  labels = {
    environment = var.environment
  }
}