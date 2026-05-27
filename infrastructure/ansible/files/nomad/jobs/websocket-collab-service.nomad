# jobs/collab-service.nomad
# ==================================================================
# Nomad job for collaboration service + Redis
# ==================================================================

job "collab-service" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"

  # Rolling update strategy (collab service only; Redis is stateful)
  update {
    max_parallel     = 1
    health_check     = "task_states"
    min_healthy_time = "30s"
    auto_revert      = true
  }

  # ------------------- Redis (Stateful) -------------------
  group "redis" {
    count = 1

    network {
      mode = "bridge"
      port "redis" {
        to = 6379
      }
    }

    # Host volume for persistent Redis data
    volume "redis_data" {
      type      = "host"
      source    = "vegamaps_redis_data"   # pre‑created on each client
      read_only = false
    }

    task "redis" {
      driver = "docker"

      config {
        image = "${REDIS_IMAGE:-redis:7-alpine}"
        ports = ["redis"]
        args = [
          "redis-server",
          "--appendonly", "yes",
          "--maxmemory", "${REDIS_MAXMEMORY:-256mb}",
          "--maxmemory-policy", "${REDIS_MAXMEMORY_POLICY:-allkeys-lru}",
          "--save", "${REDIS_SAVE_SECONDS:-60}", "${REDIS_SAVE_CHANGES:-1000}"
        ]
      }

      volume_mount {
        volume      = "redis_data"
        destination = "/data"
      }

      resources {
        cpu    = 500
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
        check {
          type     = "script"
          command  = "redis-cli"
          args     = ["ping"]
          interval = "30s"
          timeout  = "5s"
        }
      }

      logs {
        max_files     = 3
        max_file_size = 10   # MB
      }
    }
  }

  # ------------------- Collaboration Service -------------------
  group "collab" {
    count = 1   # scale as needed

    network {
      mode = "bridge"
      port "http" {
        to = "${COLLAB_CONTAINER_PORT:-8090}"
        static = "${VEGAMAPS_COLLAB_HOST_PORT:-8090}"   # optional: fixed host port
      }
    }

    task "collab" {
      driver = "docker"

      config {
        image = "cr.yandex/${YANDEX_REGISTRY_ID}/vegamaps-collab-service:${IMAGE_TAG}"
        ports = ["http"]
        init  = true   # equivalent to docker `--init`
      }

      env {
        NODE_ENV          = "${NODE_ENV:-production}"
        HOST              = "${COLLAB_HOST:-0.0.0.0}"
        PORT              = "${COLLAB_CONTAINER_PORT:-8090}"
        VEGAMAPS_ROOM_PREFIX = "${VEGAMAPS_ROOM_PREFIX:-vegamaps_}"
        USE_REDIS         = "${USE_REDIS:-true}"
        REDIS_HOST        = "redis.service.consul"   # using Consul DNS
        REDIS_PORT        = "${REDIS_PORT:-6379}"
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      # Service registration (for discovery by other services, e.g., nginx)
      service {
        name = "collab"
        port = "http"
        tags = ["collaboration", "production"]
        check {
          type     = "http"
          path     = "/health"          # adjust to your app's health endpoint
          interval = "30s"
          timeout  = "5s"
        }
      }

      logs {
        max_files     = 3
        max_file_size = 10
      }
    }
  }
}