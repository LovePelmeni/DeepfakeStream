# Look up OS image
data "yandex_compute_image" "os_image" {
  family = var.os_image_family
}

# Main compute instance
resource "yandex_compute_instance" "this" {
  name        = var.instance_name
  folder_id   = var.folder_id
  zone        = var.zone
  platform_id = var.platform_id

  resources {
    cores         = var.cores
    core_fraction = var.core_fraction
    memory        = var.ram_bytes
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
    var.user_data != "" ? { user-data = var.user_data } : {}
  )

  lifecycle {
    create_before_destroy = true
  }

  labels = {
    environment = var.environment
    purpose     = "frontend"
  }
}