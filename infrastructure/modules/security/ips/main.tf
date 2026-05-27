# Reserve a static public IP for each instance
resource "yandex_vpc_address" "static_ip" {
  for_each = var.instances
  name     = "${each.key}-ip"
  external_ipv4_address {
    zone_id = var.yc_zone
  }
}

# Create compute instances
resource "yandex_compute_instance" "soc_vm" {
  for_each    = var.instances
  name        = each.key
  platform_id = "standard-v4"
  zone        = var.yc_zone

  resources {
    cores  = each.value.cpu
    memory = each.value.ram
  }

  boot_disk {
    initialize_params {
      image_id = var.image_id
      size     = each.value.disk
    }
  }

  network_interface {
    subnet_id = var.subnet_id
    nat       = true   # ephemeral IP for initial access (but static IP overrides)
    ip_address = yandex_vpc_address.static_ip[each.key].external_ipv4_address[0].address
  }

  metadata = {
    ssh-keys = "${var.ssh_user}:${file(var.ssh_public_key)}"
  }
}