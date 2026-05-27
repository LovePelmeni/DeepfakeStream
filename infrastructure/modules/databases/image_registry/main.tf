# Create the container registry
resource "yandex_container_registry" "docker_container_registry" {
  name   = var.registry_name
  labels = var.labels
  folder_id = var.folder_id
}

# Create repositories for each service
resource "yandex_container_repository" "container_repositories" {
  for_each = var.services
  name = "${yandex_container_registry.docker_container_registry.id}/${each.key}"
}