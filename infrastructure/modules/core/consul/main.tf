data "yandex_compute_image" "os_image" {
  family = var.os_family
}

locals {
  available_zones = keys(var.private_subnet_ids)
}

resource "yandex_compute_instance" "this" {
  count = var.instance_count

  name        = "${var.environment}-consul-${count.index + 1}"
  hostname    = "consul-${count.index + 1}.${var.domain}"
  folder_id   = var.folder_id
  zone        = var.zones[count.index % length(var.zones)]
  platform_id = var.instance_platform_id

  resources {
    cores         = var.instance_cores
    memory        = var.instance_ram_gb * 1024 * 1024 * 1024
    core_fraction = var.instance_core_fraction
  }

  boot_disk {
    initialize_params {
      image_id = var.image_id != "" ? var.image_id : data.yandex_compute_image.os_image.id
      size     = var.instance_disk_size_gb
      type     = var.instance_disk_type
    }
  }

  network_interface {
    # Use the local.available_zones (which is keys of the subnet map)
    subnet_id          = var.private_subnet_ids[local.available_zones[count.index % length(local.available_zones)]]
    security_group_ids = [var.security_group_id]
    nat                = false
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
    service     = "consul"
  }
}

# Optional target group
resource "yandex_lb_target_group" "this" {
  count = var.create_target_group && var.instance_count > 0 ? 1 : 0

  name      = "${var.environment}-consul-tg"
  folder_id = var.folder_id

  dynamic "target" {
    for_each = yandex_compute_instance.this
    content {
      subnet_id = var.private_subnet_ids[target.value.zone]
      address   = target.value.network_interface[0].ip_address
    }
  }
}