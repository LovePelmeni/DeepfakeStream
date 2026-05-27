# Look up OS image
data "yandex_compute_image" "os_image" {
  family = var.os_image_family
}

# Compute instances
resource "yandex_compute_instance" "this" {
  count = var.instance_count

  name        = "${var.instance_name_prefix}-${count.index + 1}"
  folder_id   = var.folder_id
  zone        = var.zones[count.index % length(var.zones)]

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

  metadata = {
    ssh-keys = "${var.ssh_user}:${var.ssh_public_key}"
  }

  lifecycle {
    create_before_destroy = true
  }

  labels = {
    environment = var.environment
    purpose     = var.purpose_label
  }
}

# Optional target group for load balancing
resource "yandex_lb_target_group" "this" {
  count = var.create_target_group && var.instance_count > 0 ? 1 : 0

  name      = "${var.instance_name_prefix}-tg"
  folder_id = var.folder_id

  dynamic "target" {
    for_each = yandex_compute_instance.this
    content {
      subnet_id = var.subnet_id
      address   = target.value.network_interface[0].ip_address
    }
  }

  labels = {
    environment = var.environment
  }
}