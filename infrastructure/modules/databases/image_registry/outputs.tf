output "registry_id" {
  description = "ID of the created container registry"
  value       = yandex_container_registry.this.id
}

output "registry_name" {
  description = "Name of the created container registry"
  value       = yandex_container_registry.this.name
}

output "repository_ids" {
  description = "Map of repository IDs by service name"
  value = {
    for k, v in yandex_container_repository.services : k => v.id
  }
}

output "repository_names" {
  description = "Map of repository full names by service name"
  value = {
    for k, v in yandex_container_repository.services : k => v.name
  }
}

output "registry_endpoint" {
  description = "Endpoint URL for the registry"
  value       = "cr.yandex/${yandex_container_registry.this.id}"
}