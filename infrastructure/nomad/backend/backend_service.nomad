job "vegamaps-backend" {
  datacenters = ["dc1"]
  type        = "service"

  # Canary deployment – replaces blue/green logic
  update {
    canary       = 1           # number of canary instances
    max_parallel = 1           # how many to update at once
    auto_revert  = true        # auto-rollback if canary fails
    auto_promote = false       # set to true for automatic promotion after health checks
    health_check = "checks"    # use task health checks to determine success
  }

  # Variables – provide these via -var or .vars file
  variables {

    registry             = "cr.yandex/crphosdcbn5n6uhbpgi0"

    # PostgreSQL
    pg_user     = "vegamaps_user"
    pg_password = ""            # REQUIRED
    pg_name     = "gis_db"
    pg_host_port = 5439

    # Redis
    redis_password = "redispass123"
    redis_port     = 6379

    # Backend
    backend_image     = "${var.registry}/vegamaps-backend-service:stable"   # change to your image
    backend_host_port = 8000
    environment       = "production"
    jwt_secret        = ""        # REQUIRED
    jwt_algorithm     = "HS256"
    jwt_access_expire_minutes = 15
    jwt_refresh_expire_days   = 7

    # Yandex OAuth
    yandex_oauth_client_id     = ""
    yandex_oauth_client_secret = ""

    # S3 / Yandex Cloud
    s3_endpoint_url           = "https://storage.yandexcloud.net"
    s3_access_key             = ""
    s3_secret_key             = ""
    s3_bucket_name            = "vegamaps-prod"
    aws_region                = "ru-central1"
    yandex_cloud_oauth_token  = ""
    yandex_access_key_id      = ""
    yandex_secret_access_key  = ""

    # Email
    email_from      = "noreply@vegamaps.com"
    email_from_name = "VegaMaps Team"
    support_email   = "support@vegamaps.com"
    company_name    = "VegaMaps"

    # Frontend & API URLs
    frontend_url = "https://vegamaps.com"
    app_url      = "https://api.vegamaps.com"

    # Pyroscope (if used)
    pyroscope_enabled         = "true"
    pyroscope_server_address  = "http://pyroscope:4040"
    pyroscope_app_name        = "vegamaps-backend"
    pyroscope_sample_rate     = "100"

    # Health check & restart settings
    backend_health_interval = "10s"
    backend_health_timeout  = "5s"
    backend_health_retries  = 30
    backend_start_period    = "180s"

    # Nginx
    nginx_host_port = 80

    # Secrets file path for postbox private key (mounted from host)
    postbox_private_key_path = "/opt/nomad/secrets/postbox_private_key.pem"
  }

  group "vegamaps" {
    # Shared network – all tasks can communicate via localhost
    network {
      mode = "bridge"
      port "postgres" {
        static = var.pg_host_port
        to     = 5432
      }
      port "redis" {
        static = var.redis_port
        to     = 6379
      }
      port "backend" {
        static = var.backend_host_port
        to     = 8000
      }
      port "nginx" {
        static = var.nginx_host_port
        to     = 80
      }
    }

    # ---------- PostgreSQL ----------
    task "postgres" {
      driver = "docker"

      config {
        image = "postgres:15"
        ports = ["postgres"]
        volumes = [
          "/opt/nomad/volumes/vegamaps/postgres_data:/var/lib/postgresql/data"
        ]
      }

      env {
        POSTGRES_USER     = var.pg_user
        POSTGRES_PASSWORD = var.pg_password
        POSTGRES_DB       = var.pg_name
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "postgres"
        port = "postgres"
        check {
          type     = "tcp"
          interval = "10s"
          timeout  = "5s"
        }
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "fail"
      }
    }

    # ---------- Redis ----------
    task "redis" {
      driver = "docker"

      config {
        image = "redis:7-alpine"
        ports = ["redis"]
        command = "redis-server"
        args = [
          "--appendonly", "yes",
          "--requirepass", var.redis_password
        ]
        volumes = [
          "/opt/nomad/volumes/vegamaps/redis_data:/data"
        ]
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

    # ---------- Backend API (with canary support) ----------
    task "backend" {
      driver = "docker"

      config {
        image = var.backend_image
        ports = ["backend"]
        secrets = [
          {
            name   = "postbox_private_key"
            source = var.postbox_private_key_path
          }
        ]
      }

      env {
        API_VERSION                     = "v1"
        DEPLOYMENT_COLOR                = ""   # not needed
        ENVIRONMENT                     = var.environment
        DB_MODE                         = "postgres"
        PG_HOST                         = "localhost"
        PG_PORT                         = "5432"
        PG_USER                         = var.pg_user
        PG_PASSWORD                     = var.pg_password
        PG_NAME                         = var.pg_name
        REDIS_URL                       = "redis://:${var.redis_password}@localhost:6379/0"
        JWT_SECRET                      = var.jwt_secret
        JWT_ALGORITHM                   = var.jwt_algorithm
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES = var.jwt_access_expire_minutes
        JWT_REFRESH_TOKEN_EXPIRE_DAYS   = var.jwt_refresh_expire_days
        YANDEX_OAUTH_CLIENT_ID          = var.yandex_oauth_client_id
        YANDEX_OAUTH_CLIENT_SECRET      = var.yandex_oauth_client_secret
        YANDEX_OAUTH_SCOPES             = "openid email profile"
        FRONTEND_URL                    = var.frontend_url
        APP_URL                         = var.app_url
        S3_ENDPOINT_URL                 = var.s3_endpoint_url
        S3_ACCESS_KEY                   = var.s3_access_key
        S3_SECRET_KEY                   = var.s3_secret_key
        S3_BUCKET_NAME                  = var.s3_bucket_name
        AWS_DEFAULT_REGION              = var.aws_region
        YANDEX_CLOUD_OAUTH_TOKEN        = var.yandex_cloud_oauth_token
        YANDEX_ACCESS_KEY_ID            = var.yandex_access_key_id
        YANDEX_SECRET_ACCESS_KEY        = var.yandex_secret_access_key
        YANDEX_REGION                   = var.aws_region
        EMAIL_FROM                      = var.email_from
        EMAIL_FROM_NAME                 = var.email_from_name
        SUPPORT_EMAIL                   = var.support_email
        COMPANY_NAME                    = var.company_name
        PYROSCOPE_ENABLED               = var.pyroscope_enabled
        PYROSCOPE_SERVER_ADDRESS        = var.pyroscope_server_address
        PYROSCOPE_APPLICATION_NAME      = var.pyroscope_app_name
        PYROSCOPE_SAMPLE_RATE           = var.pyroscope_sample_rate
        POSTBOX_PRIVATE_KEY_PATH        = "/secrets/postbox_private_key"
      }

      resources {
        cpu    = 1000
        memory = 2048
      }

      service {
        name = "vegamaps-backend"
        port = "backend"
        check {
          type     = "http"
          path     = "/health/live"
          interval = var.backend_health_interval
          timeout  = var.backend_health_timeout
          check_restart {
            limit = var.backend_health_retries
            grace = var.backend_start_period
            ignore_warnings = false
          }
        }
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "fail"
      }
    }

    # ---------- Nginx (reverse proxy) ----------
    task "nginx" {
      driver = "docker"

      config {
        image = "vegamaps-nginx-service:latest"
        ports = ["nginx"]
        # Optionally mount nginx config if you don't bake it into the image
        # volumes = ["/opt/nomad/config/nginx/nginx.conf:/etc/nginx/nginx.conf:ro"]
      }

      env {
        ACTIVE_BACKEND_CONTAINER = "vegamaps-backend"   # The service name in Nomad
        # Nginx will need to resolve the backend – we use localhost:8000 because they share network
        # Alternatively, use a variable to pass the backend address.
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
          path     = "/"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }
}