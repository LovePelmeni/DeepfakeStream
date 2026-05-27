output "vpc_id" {
  description = "VPC network ID"
  value       = yandex_vpc_network.main.id
}

output "nat_gateway_id" {
  description = "NAT gateway ID"
  value       = yandex_vpc_gateway.nat_gateway.id
}

output "nat_route_table_id" {
  description = "NAT route table ID"
  value       = yandex_vpc_route_table.nat_route_table.id
}

output "private_subnet_ids" {
  description = "Private subnet IDs (zone => id map)"
  value = { 
    for k, v in yandex_vpc_subnet.private_sb : k => v.id 
  }
}

output "public_subnet_ids" {
  description = "Public subnet IDs (zone => id map)"
  value = { 
    for k, v in yandex_vpc_subnet.public_sb : k => v.id 
  }
}