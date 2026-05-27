data "yandex_resourcemanager_folder" "current" {
  folder_id = var.folder_id
}

resource "yandex_storage_bucket" "quarantine" {
  bucket     = "${var.name_prefix}-quarantine-${data.yandex_resourcemanager_folder.current.id}"
  folder_id  = var.folder_id
  acl        = "private"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    id      = "cleanup-quarantine"
    enabled = true
    expiration {
      days = var.quarantine_retention_days
    }
  }
  
  tags = merge(var.labels, {
    purpose = "antivirus-quarantine"
  })
}

resource "yandex_function" "scanner" {
  name               = "${var.name_prefix}-scanner"
  folder_id          = var.folder_id
  description        = "Antivirus scanning function"
  runtime            = "python312"
  entrypoint         = "handler.handle"
  memory             = var.function_memory_mb
  execution_timeout  = var.function_timeout_seconds
  
  user_hash = timestamp()
  
  content {
    zip_filename = "${path.module}/functions/scanner.zip"
  }
  
  environment = {
    QUARANTINE_BUCKET = yandex_storage_bucket.quarantine.bucket
    INFECTED_ACTION   = var.infected_files_action
  }
  
  labels = merge(var.labels, {
    service = "antivirus"
  })
}

resource "yandex_function_trigger" "upload_trigger" {
  count = var.scan_on_upload ? length(var.scan_buckets) : 0
  
  name        = "${var.name_prefix}-trigger-${count.index}"
  folder_id   = var.folder_id
  description = "Trigger antivirus scan on file upload"
  
  object_storage {
    bucket_id    = var.scan_buckets[count.index]
    create       = true
    batch_cutoff = 10
  }
  
  function {
    id             = yandex_function.scanner.id
    retry_attempts = 3
    retry_interval = 10
  }
}
