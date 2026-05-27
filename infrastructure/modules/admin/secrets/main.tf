locals {
  prefix = var.name_prefix != "" ? "${var.name_prefix}-" : ""
}

resource "yandex_kms_symmetric_key" "this" {
  folder_id           = var.folder_id
  name                = "${local.prefix}${var.kms_key_name}"
  description         = var.kms_description
  default_algorithm   = "AES_256"
  rotation_period     = var.kms_rotation_period
  deletion_protection = var.kms_deletion_protection
  labels              = var.labels
}

resource "yandex_kms_symmetric_key_iam_binding" "sa" {
  symmetric_key_id = yandex_kms_symmetric_key.this.id
  role             = "kms.admin"
  members          = ["serviceAccount:${var.secrets_manager_sa_id}"]
}

resource "yandex_lockbox_secret" "this" {
  folder_id           = var.folder_id
  name                = "${local.prefix}${var.lockbox_secret_name}"
  description         = var.lockbox_description
  deletion_protection = var.lockbox_deletion_protection
  labels              = var.labels
}

resource "yandex_lockbox_secret_iam_binding" "sa" {
  secret_id = yandex_lockbox_secret.this.id
  role      = "lockbox.admin"
  members   = ["serviceAccount:${var.secrets_manager_sa_id}"]
}

# Simple secret version - no dynamic blocks
resource "yandex_lockbox_secret_version" "initial" {
  count = length(var.lockbox_version_payload) > 0 ? 1 : 0

  secret_id   = yandex_lockbox_secret.this.id
  description = "Initial version"
}
