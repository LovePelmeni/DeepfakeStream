terraform {
  required_providers {
    yandex = {
      source = "yandex-cloud/yandex"
      # Optional: pin a version to avoid unexpected upgrades
      version = "~> 0.188"
    }
  }
  required_version = ">= 0.13"
}