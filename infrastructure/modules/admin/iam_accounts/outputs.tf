locals {
  all_service_accounts = {
    security_scanner      = yandex_iam_service_account.security_scanner.id
    bucket_fixer          = yandex_iam_service_account.bucket_fixer.id
    vm_locker             = yandex_iam_service_account.vm_locker.id
    notifier              = yandex_iam_service_account.notifier.id
    tracker               = yandex_iam_service_account.tracker.id
    response_orchestrator = yandex_iam_service_account.response_orchestrator.id
    security_deck_scanner = yandex_iam_service_account.security_deck_scanner.id
    observability         = yandex_iam_service_account.observability.id
    backend               = yandex_iam_service_account.backend.id
    ml                    = yandex_iam_service_account.ml.id
    frontend              = yandex_iam_service_account.frontend.id
    websocket_collab      = yandex_iam_service_account.websocket_collab.id
    s3_access             = yandex_iam_service_account.s3_access.id
    secrets_manager       = yandex_iam_service_account.secrets_manager.id
    network_manager       = yandex_iam_service_account.network_manager.id
    consul_server         = yandex_iam_service_account.consul_server.id
    consul_sidecar        = yandex_iam_service_account.consul_sidecar.id
  }
}

output "all_service_account_ids" {
  description = "Map of all service account IDs"
  value       = local.all_service_accounts
}

output "security_scanner_id" {
  value = yandex_iam_service_account.security_scanner.id
}

output "bucket_fixer_id" {
  value = yandex_iam_service_account.bucket_fixer.id
}

output "vm_locker_id" {
  value = yandex_iam_service_account.vm_locker.id
}

output "notifier_id" {
  value = yandex_iam_service_account.notifier.id
}

output "tracker_id" {
  value = yandex_iam_service_account.tracker.id
}

output "response_orchestrator_sa_id" {
  description = "ID of the response orchestrator service account"
  value       = yandex_iam_service_account.response_orchestrator.id
}

output "security_deck_scanner_id" {
  value = yandex_iam_service_account.security_deck_scanner.id
}

# Observability service account output
output "observability_sa_id" {
  description = "ID of the observability service account"
  value       = yandex_iam_service_account.observability.id
}

output "backend_sa_id" {
  description = "ID of the backend service account"
  value       = yandex_iam_service_account.backend.id
}

output "ml_sa_id" {
  description = "ID of the ML service account"
  value       = yandex_iam_service_account.ml.id
}

output "frontend_sa_id" {
  description = "ID of the frontend service account"
  value       = yandex_iam_service_account.frontend.id
}

output "websocket_collab_sa_id" {
  description = "ID of the WebSocket collaboration service account"
  value       = yandex_iam_service_account.websocket_collab.id
}

output "s3_access_sa_id" {
  description = "ID of the S3 access service account"
  value       = yandex_iam_service_account.s3_access.id
}

output "secrets_manager_sa_id" {
  description = "ID of the secrets management service account"
  value       = yandex_iam_service_account.secrets_manager.id
}

output "network_manager_sa_id" {
  description = "ID of the network management service account"
  value       = yandex_iam_service_account.network_manager.id
}

# ============================================
# CONSUL IAM SERVICE ACCOUNT OUTPUTS
# ============================================

output "consul_server_sa_id" {
  description = "ID of the Consul server service account"
  value       = yandex_iam_service_account.consul_server.id
}

output "consul_sidecar_sa_id" {
  description = "ID of the Consul sidecar service account"
  value       = yandex_iam_service_account.consul_sidecar.id
}

output "consul_backup_sa_id" {
  description = "ID of the Consul backup service account"
  value       = var.enable_consul_backups ? yandex_iam_service_account.consul_backup[0].id : null
}

output "consul_monitoring_sa_id" {
  description = "ID of the Consul monitoring service account"
  value       = var.enable_consul_monitoring ? yandex_iam_service_account.consul_monitoring[0].id : null
}

# ============================================
# CONSUL SERVICE ACCOUNTS SUMMARY (FIXED - removed email)
# ============================================

output "consul_service_accounts_summary" {
  description = "Summary of Consul service accounts and their purposes"
  value = {
    server = {
      id          = yandex_iam_service_account.consul_server.id
      description = "Consul server cluster operations"
      permissions = ["compute.viewer", "vpc.viewer", "load-balancer.viewer", "dns.editor", "kms.keys.encrypterDecrypter"]
    }
    sidecar = {
      id          = yandex_iam_service_account.consul_sidecar.id
      description = "Consul sidecar proxies on application VMs"
      permissions = ["compute.viewer", "vpc.viewer", "load-balancer.viewer"]
    }
    backup = {
      id          = var.enable_consul_backups ? yandex_iam_service_account.consul_backup[0].id : null
      description = "Consul state backups to S3"
      permissions = var.enable_consul_backups ? ["storage.editor"] : []
    }
    monitoring = {
      id          = var.enable_consul_monitoring ? yandex_iam_service_account.consul_monitoring[0].id : null
      description = "Consul metrics scraping and monitoring"
      permissions = var.enable_consul_monitoring ? ["monitoring.editor", "compute.viewer"] : []
    }
  }
}

# ============================================
# REGISTRY OUTPUTS
# ============================================

output "registry_pusher_account_ids" {
  description = "IDs of registry pusher service accounts by service"
  value = {
    for k, v in yandex_iam_service_account.registry_pusher : k => v.id
  }
}

output "registry_admin_account_id" {
  description = "ID of registry admin service account (if created)"
  value       = var.create_registry_admin ? yandex_iam_service_account.registry_admin[0].id : null
}

output "app_service_accounts_with_registry_pull" {
  description = "List of application service account IDs that have registry pull permissions"
  value = [
    yandex_iam_service_account.backend.id,
    yandex_iam_service_account.frontend.id,
    yandex_iam_service_account.ml.id,
    yandex_iam_service_account.websocket_collab.id
  ]
}

output "registry_permissions_summary" {
  description = "Summary of registry IAM permissions"
  value = {
    pushers = keys(yandex_iam_service_account.registry_pusher)
    pullers = [
      "backend",
      "frontend",
      "ml",
      "websocket_collab"
    ]
    admins = var.create_registry_admin ? ["registry_admin"] : []
    security_scanners = [
      "security_scanner",
      "security_deck_scanner"
    ]
    incident_response = ["response_orchestrator"]
    observability     = ["observability"]
    consul_services   = ["consul_server", "consul_sidecar"]
  }
}

output "registry_ci_cd_config" {
  description = "Configuration for CI/CD pipelines to access registry"
  value = {
    backend = {
      service_account_id = yandex_iam_service_account.registry_pusher["backend"].id
      repository         = "backend"
    }
    frontend = {
      service_account_id = yandex_iam_service_account.registry_pusher["frontend"].id
      repository         = "frontend"
    }
    ml = {
      service_account_id = yandex_iam_service_account.registry_pusher["ml"].id
      repository         = "ml"
    }
    websocket_collab = {
      service_account_id = yandex_iam_service_account.registry_pusher["websocket_collab"].id
      repository         = "websocket-collab"
    }
  }
}

output "registry_environment_variables" {
  description = "Environment variables for applications to access registry"
  value = {
    BACKEND_REGISTRY_PUSHER_SA  = yandex_iam_service_account.registry_pusher["backend"].id
    FRONTEND_REGISTRY_PUSHER_SA = yandex_iam_service_account.registry_pusher["frontend"].id
    ML_REGISTRY_PUSHER_SA       = yandex_iam_service_account.registry_pusher["ml"].id
    WS_REGISTRY_PUSHER_SA       = yandex_iam_service_account.registry_pusher["websocket_collab"].id
    OBSERVABILITY_SA_ID         = yandex_iam_service_account.observability.id
    CONSUL_SERVER_SA_ID         = yandex_iam_service_account.consul_server.id
    CONSUL_SIDECAR_SA_ID        = yandex_iam_service_account.consul_sidecar.id
  }
  sensitive = false
}

output "registry_iam_bindings" {
  description = "Details of IAM bindings created for registry access"
  value = {
    folder_level_pullers    = "All application service accounts have container-registry.images.puller"
    folder_level_pushers    = "Each service has dedicated pusher accounts with container-registry.images.pusher"
    response_orchestrator   = "container-registry.editor granted to response orchestrator"
    security_scanners       = "container-registry.viewer granted to security scanner and deck scanner"
    observability           = "monitoring.viewer, logging.reader, compute.viewer, storage.viewer granted to observability"
    consul_services         = "Consul server and sidecar service accounts have container-registry.images.puller"
  }
}

# ============================================
# CONSUL IAM BINDINGS OUTPUT
# ============================================

output "consul_iam_bindings" {
  description = "IAM bindings created for Consul service accounts"
  value = {
    consul_server_roles = [
      "compute.viewer",
      "vpc.viewer",
      "load-balancer.viewer",
      "dns.editor",
      "kms.keys.encrypterDecrypter"
    ]
    consul_sidecar_roles = [
      "compute.viewer",
      "vpc.viewer",
      "load-balancer.viewer"
    ]
    consul_backup_roles     = var.enable_consul_backups ? ["storage.editor"] : []
    consul_monitoring_roles = var.enable_consul_monitoring ? ["monitoring.editor", "compute.viewer"] : []
  }
}

# ============================================
# COMPLETE SERVICE ACCOUNTS SUMMARY (FIXED - removed email)
# ============================================

output "all_service_accounts_summary" {
  description = "Complete summary of all service accounts with their roles"
  value = {
    security = {
      security_scanner = {
        id    = yandex_iam_service_account.security_scanner.id
        roles = ["logging.viewer", "dspm.worker", "iam.viewer", "compute.viewer", "container-registry.viewer"]
      }
      bucket_fixer = {
        id    = yandex_iam_service_account.bucket_fixer.id
        roles = ["storage.editor"]
      }
      vm_locker = {
        id    = yandex_iam_service_account.vm_locker.id
        roles = ["compute.editor"]
      }
      notifier = {
        id    = yandex_iam_service_account.notifier.id
        roles = ["logging.viewer"]
      }
      tracker = {
        id    = yandex_iam_service_account.tracker.id
        roles = []
      }
      response_orchestrator = {
        id    = yandex_iam_service_account.response_orchestrator.id
        roles = ["storage.editor", "compute.editor", "vpc.admin", "logging.viewer", "container-registry.editor"]
      }
      security_deck_scanner = {
        id    = yandex_iam_service_account.security_deck_scanner.id
        roles = ["compute.viewer", "storage.viewer", "vpc.viewer", "iam.viewer", "container-registry.viewer"]
      }
    }
    observability = {
      id    = yandex_iam_service_account.observability.id
      roles = ["monitoring.viewer", "logging.reader", "compute.viewer", "storage.viewer", "kms.keys.encrypterDecrypter"]
    }
    applications = {
      backend = {
        id    = yandex_iam_service_account.backend.id
        roles = ["container-registry.images.puller"]
      }
      frontend = {
        id    = yandex_iam_service_account.frontend.id
        roles = ["container-registry.images.puller"]
      }
      ml = {
        id    = yandex_iam_service_account.ml.id
        roles = ["container-registry.images.puller"]
      }
      websocket_collab = {
        id    = yandex_iam_service_account.websocket_collab.id
        roles = ["container-registry.images.puller"]
      }
      s3_access = {
        id    = yandex_iam_service_account.s3_access.id
        roles = []
      }
    }
    infrastructure = {
      secrets_manager = {
        id    = yandex_iam_service_account.secrets_manager.id
        roles = ["kms.admin", "lockbox.admin"]
      }
      network_manager = {
        id    = yandex_iam_service_account.network_manager.id
        roles = ["vpc.admin"]
      }
    }
    service_mesh = {
      consul_server = {
        id    = yandex_iam_service_account.consul_server.id
        roles = ["compute.viewer", "vpc.viewer", "load-balancer.viewer", "dns.editor", "kms.keys.encrypterDecrypter", "container-registry.images.puller"]
      }
      consul_sidecar = {
        id    = yandex_iam_service_account.consul_sidecar.id
        roles = ["compute.viewer", "vpc.viewer", "load-balancer.viewer", "container-registry.images.puller"]
      }
      consul_backup = var.enable_consul_backups ? {
        id    = yandex_iam_service_account.consul_backup[0].id
        roles = ["storage.editor"]
      } : null
      consul_monitoring = var.enable_consul_monitoring ? {
        id    = yandex_iam_service_account.consul_monitoring[0].id
        roles = ["monitoring.editor", "compute.viewer"]
      } : null
    }
  }
}

# ============================================
# SERVICE ACCOUNT ENVIRONMENT VARIABLES REFERENCE
# ============================================

output "service_accounts_env_vars" {
  description = "Environment variable names for service accounts (for use in Terraform or CI/CD)"
  value = {
    TF_VAR_security_scanner_sa_id       = yandex_iam_service_account.security_scanner.id
    TF_VAR_bucket_fixer_sa_id           = yandex_iam_service_account.bucket_fixer.id
    TF_VAR_vm_locker_sa_id              = yandex_iam_service_account.vm_locker.id
    TF_VAR_notifier_sa_id               = yandex_iam_service_account.notifier.id
    TF_VAR_tracker_sa_id                = yandex_iam_service_account.tracker.id
    TF_VAR_response_orchestrator_sa_id  = yandex_iam_service_account.response_orchestrator.id
    TF_VAR_observability_sa_id          = yandex_iam_service_account.observability.id
    TF_VAR_backend_sa_id                = yandex_iam_service_account.backend.id
    TF_VAR_ml_sa_id                     = yandex_iam_service_account.ml.id
    TF_VAR_frontend_sa_id               = yandex_iam_service_account.frontend.id
    TF_VAR_websocket_collab_sa_id       = yandex_iam_service_account.websocket_collab.id
    TF_VAR_consul_server_sa_id          = yandex_iam_service_account.consul_server.id
    TF_VAR_consul_sidecar_sa_id         = yandex_iam_service_account.consul_sidecar.id
  }
  sensitive = false
}