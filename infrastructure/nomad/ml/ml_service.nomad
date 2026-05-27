job "vegamaps-ml" {
  datacenters = ["dc1"]
  type        = "service"

  # Canary / rolling deployment – replaces blue/green containers
  update {
    canary       = 1            # one canary instance during update
    max_parallel = 1
    auto_revert  = true         # rollback if canary fails
    auto_promote = false        # manual promotion (set true for auto)
    health_check = "checks"
  }

  variables {
    # Registry & images
    registry             = "cr.yandex/crphosdcbn5n6uhbpgi0"
    service_image_suffix = "service"   # or empty
    service_name         = "ml"        # used in container names and paths

    # Versions – change these to trigger a rolling update
    ml_version     = "latest"   # image tag for ML service
    worker_version = "latest"   # image tag for Celery worker

    # Ports (host)
    ml_port        = 8001       # host port for ML service
    nginx_port     = 80
    redis_port     = 6379
    rabbitmq_port  = 5672
    rabbitmq_ui    = 15672

    # Redis
    redis_password = "redispass123"

    # RabbitMQ
    rabbitmq_user  = "vegamaps"
    rabbitmq_pass  = "vegamaps123"
    rabbitmq_vhost = "vegamaps"

    # Database (if your ML service uses a DB – optional)
    db_host     = "postgres"    # or another service
    db_port     = 5432
    db_name     = "vegamaps_production"
    db_user     = "vegamaps"
    db_password = ""            # REQUIRED if used

    # Yandex S3 / Cloud
    yandex_s3_access_key = ""
    yandex_s3_secret_key = ""
    yc_s3_bucket         = "vegamaps-models"
    yc_s3_endpoint       = "storage.yandexcloud.net"
    kms_key_id           = ""

    # Observability (optional)
    otel_exporter_otlp_endpoint = ""
    evidently_collector_url     = ""

    # Application
    api_version      = "v1"
    environment      = "production"
    log_level        = "INFO"
    workers          = 4
    health_path      = "health"          # endpoint without /api/v1/ prefix
    health_interval  = "30s"
    health_timeout   = "10s"
    health_retries   = 3
    health_start_period = "60s"

    # Celery worker – enable/disable
    enable_celery_worker = true

    # Host volume paths (create these directories on Nomad client)
    redis_data_volume     = "/opt/nomad/volumes/vegamaps/redis_data"
    rabbitmq_data_volume  = "/opt/nomad/volumes/vegamaps/rabbitmq_data"
    ml_weights_volume     = "/opt/nomad/volumes/vegamaps/ml_weights"
    ml_service_data       = "/opt/nomad/volumes/vegamaps/ml_service_data"
    shared_temp_volume    = "/opt/nomad/volumes/vegamaps/shared_temp"
  }

  group "ml" {
    network {
      mode = "bridge"
      port "ml" {
        static = var.ml_port
        to     = 8000
      }
      port "nginx" {
        static = var.nginx_port
        to     = 80
      }
      port "redis" {
        static = var.redis_port
        to     = 6379
      }
      port "rabbitmq" {
        static = var.rabbitmq_port
        to     = 5672
      }
      port "rabbitmq_ui" {
        static = var.rabbitmq_ui
        to     = 15672
      }
    }

    # ---------- Redis ----------
    task "redis" {
      driver = "docker"
      config {
        image = "redis:7-alpine"
        ports = ["redis"]
        args = [
          "redis-server", "--appendonly", "yes",
          "--requirepass", var.redis_password
        ]
        volumes = ["${var.redis_data_volume}:/data"]
      }
      env {
        REDIS_PASSWORD = var.redis_password
      }
      resources {
        cpu    = 200
        memory = 512
      }
      service {
        name = "redis"
        port = "redis"
        check {
          type     = "tcp"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- RabbitMQ ----------
    task "rabbitmq" {
      driver = "docker"
      config {
        image = "rabbitmq:3-management-alpine"
        ports = ["rabbitmq", "rabbitmq_ui"]
        volumes = ["${var.rabbitmq_data_volume}:/var/lib/rabbitmq"]
      }
      env {
        RABBITMQ_DEFAULT_USER = var.rabbitmq_user
        RABBITMQ_DEFAULT_PASS = var.rabbitmq_pass
        RABBITMQ_DEFAULT_VHOST = var.rabbitmq_vhost
      }
      resources {
        cpu    = 500
        memory = 1024
      }
      service {
        name = "rabbitmq"
        port = "rabbitmq"
        check {
          type     = "tcp"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }

    # ---------- ML Service (main) – replaces blue/green ----------
    task "ml-service" {
      driver = "docker"
      config {
        image = "${var.registry}/vegamaps-ml-${var.service_image_suffix}:${var.ml_version}"
        ports = ["ml"]
        volumes = [
          "${var.ml_weights_volume}:/app/weights",
          "${var.ml_service_data}:/app/data",
          "${var.shared_temp_volume}:/tmp/vegamaps"
        ]
      }
      env {
        ML_SERVICE        = var.service_name
        ML_SERVICE_PORT   = var.ml_port
        API_VERSION       = var.api_version
        DEPLOYMENT_COLOR  = ""   # not needed with Nomad canary
        SERVICE_VERSION   = var.ml_version
        MODEL_VERSION     = var.ml_version
        ENVIRONMENT       = var.environment
        DB_HOST           = var.db_host
        DB_PORT           = var.db_port
        DB_NAME           = var.db_name
        DB_USER           = var.db_user
        DB_PASSWORD       = var.db_password
        REDIS_HOST        = "localhost"
        REDIS_PORT        = var.redis_port
        REDIS_PASSWORD    = var.redis_password
        RABBITMQ_HOST     = "localhost"
        RABBITMQ_PORT     = var.rabbitmq_port
        RABBITMQ_USER     = var.rabbitmq_user
        RABBITMQ_PASSWORD = var.rabbitmq_pass
        RABBITMQ_VHOST    = var.rabbitmq_vhost
        CELERY_BROKER_URL = "amqp://${var.rabbitmq_user}:${var.rabbitmq_pass}@localhost:${var.rabbitmq_port}/${var.rabbitmq_vhost}"
        CELERY_RESULT_BACKEND = "redis://:${var.redis_password}@localhost:${var.redis_port}/1"
        YANDEX_S3_ACCESS_KEY = var.yandex_s3_access_key
        YANDEX_S3_SECRET_KEY = var.yandex_s3_secret_key
        YC_S3_BUCKET          = var.yc_s3_bucket
        YC_S3_ENDPOINT        = var.yc_s3_endpoint
        KMS_KEY_ID            = var.kms_key_id
        LOG_LEVEL             = var.log_level
        OTEL_EXPORTER_OTLP_ENDPOINT = var.otel_exporter_otlp_endpoint
        EVIDENTLY_COLLECTOR_URL     = var.evidently_collector_url
        WORKERS               = var.workers
        APPLICATION_HOST      = "0.0.0.0"
      }
      resources {
        cpu    = 2000
        memory = 4096
      }
      service {
        name = "vegamaps-ml-service"
        port = "ml"
        check {
          type     = "http"
          path     = "/api/${var.api_version}/${var.health_path}"
          interval = var.health_interval
          timeout  = var.health_timeout
          check_restart {
            limit = var.health_retries
            grace = var.health_start_period
          }
        }
      }
    }

    # ---------- Celery Worker (optional) ----------
    task "celery-worker" {
      driver = "docker"
      config {
        image = "${var.registry}/vegamaps-ml-worker:${var.worker_version}"
      }
      env {
        SERVICE_NAME          = var.service_name
        SERVICE_VERSION       = var.ml_version
        MODEL_VERSION         = var.ml_version
        SERVICE_PORT          = 8000
        CELERY_BROKER_URL     = "amqp://${var.rabbitmq_user}:${var.rabbitmq_pass}@localhost:${var.rabbitmq_port}/${var.rabbitmq_vhost}"
        CELERY_RESULT_BACKEND = "redis://:${var.redis_password}@localhost:${var.redis_port}/1"
        REDIS_HOST            = "localhost"
        REDIS_PORT            = var.redis_port
        REDIS_PASSWORD        = var.redis_password
        YC_S3_BUCKET          = var.yc_s3_bucket
        YC_S3_ENDPOINT        = var.yc_s3_endpoint
        YANDEX_S3_ACCESS_KEY  = var.yandex_s3_access_key
        YANDEX_S3_SECRET_KEY  = var.yandex_s3_secret_key
        ENVIRONMENT           = var.environment
        LOG_LEVEL             = var.log_level
        APPLICATION_HOST      = "0.0.0.0"
      }
      volumes = [
        "${var.ml_weights_volume}:/app/weights",
        "${var.ml_service_data}:/app/data",
        "${var.shared_temp_volume}:/tmp/vegamaps"
      ]
      resources {
        cpu    = 1000
        memory = 2048
      }
      restart {
        attempts = 10
        interval = "5m"
        mode     = "fail"
      }
    }

    # ---------- Nginx (reverse proxy to ml-service) ----------
    task "nginx" {
      driver = "docker"
      config {
        image = "nginx:1.27-alpine"
        ports = ["nginx"]
        volumes = [
          # Mount your custom nginx config and templates from host
          "/opt/nomad/config/vegamaps/nginx/nginx.conf:/etc/nginx/nginx.conf:ro",
          "/opt/nomad/config/vegamaps/nginx/templates:/etc/nginx/templates:ro"
        ]
      }
      env {
        # Nginx will proxy to localhost:8000 (ml-service)
        UPSTREAM_HOST = "localhost"
        UPSTREAM_PORT = 8000
      }
      resources {
        cpu    = 100
        memory = 128
      }
      service {
        name = "nginx"
        port = "nginx"
        check {
          type     = "http"
          path     = "/health"   # adjust if nginx has a health endpoint
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }
}