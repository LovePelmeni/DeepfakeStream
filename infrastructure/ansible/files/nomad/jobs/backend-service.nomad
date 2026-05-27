# jobs/backend.nomad
job "backend" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"

  # Rolling update with canary (blue‑green)
  update {
    max_parallel     = 1
    health_check     = "task_states"
    min_healthy_time = "30s"
    auto_revert      = true
    canary           = 1        # one green instance during update
  }

  # PostgreSQL + Redis run together (share network)
  group "databases" {
    count = 1
    network {
      mode = "bridge"
      port "postgres" {
        to = 5432
        static = 5439   # matches ${POSTGRES_HOST_PORT:-5439}
      }
      port "redis" {
        to = 6379
        static = 6379
      }
    }

    # Persistent volumes for stateful data
    volume "postgres_data" {
      type   = "host"
      source = "vegamaps_postgres_data"   # pre‑created host volume
    }
    volume "redis_data" {
      type   = "host"
      source = "vegamaps_redis_data"
    }

    task "postgres" {
      driver = "docker"
      config {
        image = "postgres:15"
        ports = ["postgres"]
      }
      env {
        POSTGRES_USER     = "${attr.unique.platform.aws.instance-id}"  # Use Nomad variables
        POSTGRES_PASSWORD = "{{ key "secret/postgres/password" }}"    # from Consul KV
        POSTGRES_DB       = "gis_db"
      }
      volume_mount {
        volume      = "postgres_data"
        destination = "/var/lib/postgresql/data"
      }
      resources {
        cpu    = 2000
        memory = 4096
      }
      service {
        name = "postgres"
        port = "postgres"
        check {
          type     = "script"
          command  = "/usr/local/bin/pg_isready"
          args     = ["-U", "${POSTGRES_USER}", "-d", "${POSTGRES_DB}"]
          interval = "10s"
          timeout  = "5s"
        }
      }
    }

    task "redis" {
      driver = "docker"
      config {
        image = "redis:7-alpine"
        ports = ["redis"]
        args  = [
          "redis-server",
          "--appendonly", "yes",
          "--maxmemory", "256mb",
          "--maxmemory-policy", "allkeys-lru",
          "--requirepass", "{{ key "secret/redis/password" }}"
        ]
      }
      volume_mount {
        volume      = "redis_data"
        destination = "/data"
      }
      resources {
        cpu    = 1000
        memory = 1024
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
  }

  # Backend service (supports blue‑green via canary)
  group "backend" {
    count = 1   # one stable + one canary during deploy
    network {
      mode = "bridge"
      port "http" {
        to = 8000
        static = 8000   # main port for the service (blue/green share)
      }
    }

    # Secret file (postbox private key) as a template
    template {
      data        = <<EOF
{{ with secret "secret/postbox_private_key" }}{{ .Data.value }}{{ end }}
EOF
      destination = "secrets/postbox_private_key.pem"
      perms       = "0400"
      change_mode = "restart"
    }

    task "app" {
      driver = "docker"
      config {
        image = "$cr.yandex/${YANDEX_REGISTRY_ID}/vegamaps-backend-service:${IMAGE_TAG}"
        ports = ["http"]
        # Sealed secret file mount
        volumes = [
          "secrets/postbox_private_key.pem:/run/secrets/postbox_private_key.pem"
        ]
      }
      env {
        API_VERSION                     = "v1"
        ENVIRONMENT                     = "production"
        DB_MODE                         = "postgres"
        PG_HOST                         = "localhost"
        PG_PORT                         = "5432"
        PG_USER                         = "vegamaps_user"
        PG_PASSWORD                     = "{{ key "secret/postgres/password" }}"
        PG_NAME                         = "gis_db"
        REDIS_URL                       = "redis://:{{ key "secret/redis/password" }}@localhost:6379/0"
        JWT_SECRET                      = "{{ key "secret/jwt/secret" }}"
        JWT_ALGORITHM                   = "HS256"
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES = "15"
        JWT_REFRESH_TOKEN_EXPIRE_DAYS   = "7"
        YANDEX_OAUTH_CLIENT_ID          = "{{ key "secret/oauth/client_id" }}"
        YANDEX_OAUTH_CLIENT_SECRET      = "{{ key "secret/oauth/client_secret" }}"
        YANDEX_OAUTH_SCOPES             = "openid email profile"
        FRONTEND_URL                    = "https://vegamaps.com"
        S3_ENDPOINT_URL                 = "https://storage.yandexcloud.net"
        S3_ACCESS_KEY                   = "{{ key "secret/s3/access_key" }}"
        S3_SECRET_KEY                   = "{{ key "secret/s3/secret_key" }}"
        S3_BUCKET_NAME                  = "vegamaps-prod"
        AWS_DEFAULT_REGION              = "ru-central1"
        EMAIL_FROM                      = "noreply@vegamaps.com"
        EMAIL_FROM_NAME                 = "VegaMaps Team"
        APP_URL                         = "https://api.vegamaps.com"
        SUPPORT_EMAIL                   = "support@vegamaps.com"
        COMPANY_NAME                    = "VegaMaps"
        PYROSCOPE_ENABLED               = "true"
        PYROSCOPE_SERVER_ADDRESS        = "http://pyroscope:4040"
        PYROSCOPE_SAMPLE_RATE           = "100"
        POSTBOX_PRIVATE_KEY_PATH        = "/run/secrets/postbox_private_key.pem"
      }
      resources {
        cpu    = 4000
        memory = 8192   # adjust per your needs
      }
      service {
        name = "backend"
        port = "http"
        tags = ["blue"]   # becomes "green" during canary
        check {
          type     = "http"
          path     = "/health/live"
          interval = "10s"
          timeout  = "5s"
        }
      }
    }
  }

  # NGINX router – uses Consul to find active backend (blue or green)
  group "nginx" {
    count = 1
    network {
      mode = "bridge"
      port "http" {
        to = 80
        static = 80   # externally accessible port
      }
    }

    task "nginx" {
      driver = "docker"
      config {
        image = "nginx:alpine"
        ports = ["http"]
      }
      template {
        data = <<EOF
server {
    listen 80;
    location / {
        set $backend http://backend.service.consul:8000;
        proxy_pass $backend;
        proxy_set_header Host $host;
    }
}
EOF
        destination = "/local/nginx.conf"
        change_mode = "signal"
        change_signal = "SIGHUP"
      }
      config {
        volumes = ["/local/nginx.conf:/etc/nginx/conf.d/default.conf:ro"]
      }
      resources {
        cpu    = 500
        memory = 256
      }
    }
  }
}