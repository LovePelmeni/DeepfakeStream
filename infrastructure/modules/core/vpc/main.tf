# VPC Network
resource "yandex_vpc_network" "main" {
  name       = var.net_name
  folder_id  = var.yandex_cloud_folder_id
  description = "DroneSpas VPC - ${var.net_name}"
  
  labels = {
    environment = "prod"
    project     = "dronespas"
  }
}

# NAT gateway
resource "yandex_vpc_gateway" "nat_gateway" {
  // VPC Public Internet gateway configuraiton
  name = "${var.net_name}-nat-gateway"
  shared_egress_gateway {}
}

# Route tables
resource "yandex_vpc_route_table" "nat_route_table" {
  // VPC public route table configuration
  name       = "rt-public"
  network_id = yandex_vpc_network.main.id
  static_route {
    destination_prefix = "0.0.0.0/0"
    gateway_id         = yandex_vpc_gateway.nat_gateway.id
  }
}

# Private Subnets (from var.net_private_subnets)
resource "yandex_vpc_subnet" "private_sb" {
  for_each = { for idx, subnet in var.net_private_subnets : subnet.zone => subnet }
  
  name           = "${var.net_name}-private-${each.key}"
  zone           = each.key
  network_id     = yandex_vpc_network.main.id
  v4_cidr_blocks = [each.value.cidr]
  folder_id      = var.yandex_cloud_folder_id
  route_table_id = yandex_vpc_route_table.nat_route_table.id // forwards traffic to route table
}

# Public Subnets (from var.net_public_subnets)  
resource "yandex_vpc_subnet" "public_sb" {
  for_each = { for idx, subnet in var.net_public_subnets : subnet.zone => subnet }
  
  name           = "${var.net_name}-public-${each.key}"
  zone           = each.key
  network_id     = yandex_vpc_network.main.id
  v4_cidr_blocks = [each.value.cidr]
  folder_id      = var.yandex_cloud_folder_id
  route_table_id = yandex_vpc_route_table.nat_route_table.id
}