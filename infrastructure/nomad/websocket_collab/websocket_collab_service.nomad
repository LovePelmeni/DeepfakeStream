job "vegamaps-collab" {
  datacenters = ["dc1"]
  type        = "service"

  # Optional: enable rolling updates with canary (set canary=1 if needed)
  update {
    max_parallel = 1
    canary       = 0          # set to 1 to enable canary deployments
    auto_revert  = true
    health_check = "checks"
  }

  variables {
    # Yandex registry
    registry = "cr.yandex/crphosdcbn5n6uhbpgi0"   # REQUIRED

    # Image tags
    image_tag = ""            # REQUIRED, e.g. "v1.2.3"

    # Collaboration service settings
    collab_container_port = 8090
    collab_host_port      = 8090
    collab_host           = "0.0.0.0"
    node_env              = "production"
    vegamaps_room_prefix  = "vegamaps_"
    use_redis             = "true"

    # Redis settings
    redis_host            = "localhost"   # will use localhost because same group
    redis_port            = 6379
    redis_image           = "redis:7-alpine"
    redis_maxmemory       = "256mb"
    redis_maxmemory_policy = "allkeys-lru"
    redis_save_seconds    = "60"
    redis_save_changes    = "1000"
    redis_health_interval = "30s"
    redis_health_timeout  = "5s"
    redis_health_retries  = 5
    redis_health_start_period = "20s"

    # Volume names (host paths)
    redis_volume_name = "/opt/nomad/volumes/vegamaps/redis_data"   # adjust as needed
  }

  group "collab" {
    network {
      mode = "bridge"
      port "collab" {
        static = var.collab_host_port
        to     = var.collab_container_port
      }
      # Redis does not need a host port – only internal
    }

    # ---------- Redis ----------
    task "redis" {
      driver = "docker"

      config {
        image = var.redis_image
        args = [
          "redis-server",
          "--appendonly", "yes",
          "--maxmemory", var.redis_maxmemory,
          "--maxmemory-policy", var.redis_maxmemory_policy,
          "--save", var.redis_save_seconds, var.redis_save_changes
        ]
        volumes = [
          "${var.redis_volume_name}:/data"
        ]
      }

      resources {
        cpu    = 200
        memory = 512
      }

      service {
        name = "redis"
        port = 6379
        check {
          type     = "tcp"
          interval = var.redis_health_interval
          timeout  = var.redis_health_timeout
          check_restart {
            limit = var.redis_health_retries
            grace = var.redis_health_start_period
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

    # ---------- Collaboration Service (Node.js) ----------
    task "collab" {
      driver = "docker"

      config {
        image = "${var.registry}/vegamaps-collab-service:${var.image_tag}"
        ports = ["collab"]
        init  = true   # equivalent to docker --init
      }

      env {
        NODE_ENV           = var.node_env
        HOST               = var.collab_host
        PORT               = var.collab_container_port
        VEGAMAPS_ROOM_PREFIX = var.vegamaps_room_prefix
        USE_REDIS          = var.use_redis
        REDIS_HOST         = var.redis_host   # "localhost"
        REDIS_PORT         = var.redis_port
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "vegamaps-collab"
        port = "collab"
        check {
          type     = "http"
          path     = "/health"      # adjust if your service has a health endpoint
          interval = "30s"
          timeout  = "5s"
          # If no health endpoint, use tcp check:
          # type = "tcp"
        }
      }

      restart {
        attempts = 10
        interval = "5m"
        delay    = "15s"
        mode     = "fail"
      }
    }
  }
}