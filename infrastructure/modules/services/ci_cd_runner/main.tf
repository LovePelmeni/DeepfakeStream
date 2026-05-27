# Look up OS image
data "yandex_compute_image" "os_image" {
  family = var.os_image_family
}

# Optional data disk (for build artifacts, cache, etc.)
resource "yandex_compute_disk" "data_disk" {
  count = var.data_disk_enabled ? 1 : 0

  name      = "${var.instance_name_prefix}-data-disk"
  type      = var.data_disk_type
  size      = var.data_disk_size_gb
  zone      = var.zone
  folder_id = var.folder_id

  labels = {
    environment = var.environment
    purpose     = "runner-data"
  }
}

# Main compute instance (GitHub runner)
resource "yandex_compute_instance" "this" {
  name        = "${var.instance_name_prefix}-${var.environment}"
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

  dynamic "secondary_disk" {
    for_each = var.data_disk_enabled ? [1] : []
    content {
      disk_id     = yandex_compute_disk.data_disk[0].id
      auto_delete = false
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
    purpose     = "github-runner"
  }
}