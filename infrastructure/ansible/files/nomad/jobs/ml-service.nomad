# ==================================================================
# Nomad job for a single ML service (parameterised)
# Usage: nomad job run -var="SERVICE_NAME=..." -var="..." ml-service.nomad
# ==================================================================

job "ml-${var.SERVICE_NAME}" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"

  # ------------------------------------------------------------------
  # Redis – per‑service, stateful
  # ------------------------------------------------------------------
  group "redis" {
    count = 1
    network { port "redis" { to = 6379 } }

    volume "redis_data" {
      type      = "host"
      source    = "vegamaps_redis_${var.SERVICE_NAME}"
      read_only = false
    }

    task "redis" {
      driver = "docker"
      config {
        image = "redis:7-alpine"
        ports = ["redis"]
        args = ["redis-server", "--appendonly", "yes", "--requirepass", var.redis_password]
      }
      env { REDIS_PASSWORD = var.redis_password }
      volume_mount {
        volume      = "redis_data"
        destination = "/data"
      }
      resources { cpu = 500; memory = 1024 }
      service {
        name = "redis-${var.SERVICE_NAME}"
        port = "redis"
        check {
          type     = "script"
          command  = "redis-cli"
          args     = ["-a", var.redis_password, "ping"]
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # RabbitMQ – per‑service, stateful
  # ------------------------------------------------------------------
  group "rabbitmq" {
    count = 1
    network {
      port "amqp" { to = 5672 }
      port "ui"   { to = 15672 }
    }

    volume "rabbitmq_data" {
      type      = "host"
      source    = "vegamaps_rabbitmq_${var.SERVICE_NAME}"
      read_only = false
    }

    task "rabbitmq" {
      driver = "docker"
      config {
        image = "rabbitmq:3-management-alpine"
        ports = ["amqp", "ui"]
      }
      env {
        RABBITMQ_DEFAULT_USER = var.rabbitmq_user
        RABBITMQ_DEFAULT_PASS = var.rabbitmq_password
        RABBITMQ_DEFAULT_VHOST = var.rabbitmq_vhost
      }
      volume_mount {
        volume      = "rabbitmq_data"
        destination = "/var/lib/rabbitmq"
      }
      resources { cpu = 500; memory = 1024 }
      service {
        name = "rabbitmq-${var.SERVICE_NAME}"
        port = "amqp"
        check {
          type     = "script"
          command  = "rabbitmq-diagnostics"
          args     = ["ping"]
          interval = "30s"
          timeout  = "10s"
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # ML Service (single instance, no colours)
  # ------------------------------------------------------------------
  group "ml-service" {
    count = 1
    network {
      port "http" {
        to = 8000
        static = var.ml_port   # host port (e.g., 8003)
      }
    }

    # Volumes for weights, data, temp
    volume "weights" {
      type      = "host"
      source    = "vegamaps_weights_${var.SERVICE_NAME}"
      read_only = false
    }
    volume "data" {
      type      = "host"
      source    = "vegamaps_data_${var.SERVICE_NAME}"
      read_only = false
    }
    volume "temp" {
      type      = "host"
      source    = "vegamaps_temp_${var.SERVICE_NAME}"
      read_only = false
    }

    task "ml" {
      driver = "docker"
      config {
        image = "${var.registry}/vegamaps-ml-${var.service_image_suffix}:${var.service_version}"
        ports = ["http"]
      }
      env {
        ML_SERVICE            = var.service_name
        ML_SERVICE_PORT       = var.ml_port
        API_VERSION           = var.api_version
        DEPLOYMENT_COLOR      = "single"
        SERVICE_VERSION       = var.service_version
        MODEL_VERSION         = var.model_version
        ENVIRONMENT           = var.environment
        DB_HOST               = var.db_host
        DB_PORT               = var.db_port
        DB_NAME               = var.db_name
        DB_USER               = var.db_user
        DB_PASSWORD           = var.db_password
        REDIS_HOST            = "redis"
        REDIS_PORT            = "6379"
        REDIS_PASSWORD        = var.redis_password
        RABBITMQ_HOST         = "rabbitmq"
        RABBITMQ_PORT         = "5672"
        RABBITMQ_USER         = var.rabbitmq_user
        RABBITMQ_PASSWORD     = var.rabbitmq_password
        RABBITMQ_VHOST        = var.rabbitmq_vhost
        CELERY_BROKER_URL     = "amqp://${var.rabbitmq_user}:${var.rabbitmq_password}@rabbitmq:5672/${var.rabbitmq_vhost}"
        CELERY_RESULT_BACKEND = "redis://:${var.redis_password}@redis:6379/1"
        YANDEX_S3_ACCESS_KEY  = var.yandex_s3_access_key
        YANDEX_S3_SECRET_KEY  = var.yandex_s3_secret_key
        YC_S3_BUCKET          = var.yc_s3_bucket
        YC_S3_ENDPOINT        = var.yc_s3_endpoint
        KMS_KEY_ID            = var.kms_key_id
        LOG_LEVEL             = var.log_level
        OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_endpoint
        EVIDENTLY_COLLECTOR_URL    = var.evidently_url
        WORKERS               = var.workers
        APPLICATION_HOST      = "0.0.0.0"
      }
      volume_mount {
        volume      = "weights"
        destination = "/app/weights"
      }
      volume_mount {
        volume      = "data"
        destination = "/app/data"
      }
      volume_mount {
        volume      = "temp"
        destination = "/tmp/vegamaps"
      }
      resources {
        cpu    = var.ml_cpu
        memory = var.ml_memory
      }
      service {
        name = "ml-${var.service_name}"
        port = "http"
        check {
          type     = "http"
          path     = "/api/${var.api_version}/${var.health_path}"
          interval = var.health_interval
          timeout  = var.health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Celery Worker
  # ------------------------------------------------------------------
  group "celery-worker" {
    count = var.celery_workers
    network { mode = "bridge" }

    volume "weights" {
      type      = "host"
      source    = "vegamaps_weights_${var.SERVICE_NAME}"
      read_only = false
    }
    volume "data" {
      type      = "host"
      source    = "vegamaps_data_${var.SERVICE_NAME}"
      read_only = false
    }
    volume "temp" {
      type      = "host"
      source    = "vegamaps_temp_${var.SERVICE_NAME}"
      read_only = false
    }

    task "celery" {
      driver = "docker"
      config {
        image = "${var.registry}/vegamaps-ml-worker:${var.service_version}"
      }
      env {
        SERVICE_NAME       = var.service_name
        SERVICE_VERSION    = var.service_version
        MODEL_VERSION      = var.model_version
        SERVICE_PORT       = "8000"
        CELERY_BROKER_URL  = "amqp://${var.rabbitmq_user}:${var.rabbitmq_password}@rabbitmq:5672/${var.rabbitmq_vhost}"
        CELERY_RESULT_BACKEND = "redis://:${var.redis_password}@redis:6379/1"
        REDIS_HOST         = "redis"
        REDIS_PORT         = "6379"
        REDIS_PASSWORD     = var.redis_password
        YC_S3_BUCKET       = var.yc_s3_bucket
        YC_S3_ENDPOINT     = var.yc_s3_endpoint
        YANDEX_S3_ACCESS_KEY = var.yandex_s3_access_key
        YANDEX_S3_SECRET_KEY = var.yandex_s3_secret_key
        ENVIRONMENT        = var.environment
        LOG_LEVEL          = var.log_level
      }
      volume_mount {
        volume      = "weights"
        destination = "/app/weights"
      }
      volume_mount {
        volume      = "data"
        destination = "/app/data"
      }
      volume_mount {
        volume      = "temp"
        destination = "/tmp/vegamaps"
      }
      resources {
        cpu    = var.celery_cpu
        memory = var.celery_memory
      }
    }
  }

  # ------------------------------------------------------------------
  # Nginx reverse proxy (exposes ML service)
  # ------------------------------------------------------------------
  group "nginx" {
    count = 1
    network {
      port "http" {
        to = 80
        static = var.nginx_port
      }
    }

    task "nginx" {
      driver = "docker"
      config {
        image = "nginx:1.27-alpine"
        ports = ["http"]
      }
      template {
        data = <<EOF
server {
    listen 80;
    location / {
        proxy_pass http://ml-${var.service_name}.service.consul:8000;
        proxy_set_header Host $host;
    }
    location /health {
        return 200 "healthy\\n";
    }
}
EOF
        destination = "/etc/nginx/conf.d/default.conf"
        change_mode = "signal"
        change_signal = "SIGHUP"
      }
      resources {
        cpu    = 200
        memory = 256
      }
      service {
        name = "nginx-${var.service_name}"
        port = "http"
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }
}