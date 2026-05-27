job "vegamaps-frontend" {
  datacenters = ["dc1"]
  type        = "service"

  # Canary deployment – enables blue/green style updates
  update {
    canary       = 1           # run one canary instance during update
    max_parallel = 1
    auto_revert  = true        # auto-rollback if canary fails
    auto_promote = false       # manual promotion required (set to true for automatic)
    health_check = "checks"
  }

  variables {
    # Yandex registry ID (required)
    registry = "cr.yandex/crphosdcbn5n6uhbpgi0"

    # Redis
    redis_image         = "redis:7-alpine"
    redis_restart       = "unless-stopped"
    redis_health_interval = "10s"

    # Frontend blue
    frontend_blue_tag   = ""   # REQUIRED – e.g., "v1.2.3"
    frontend_blue_image = "${var.registry}/vegamaps-frontend-service:${var.frontend_blue_tag}"

    # Frontend green
    frontend_green_tag  = ""   # REQUIRED for green deployment, can be same as blue initially
    frontend_green_image = "${var.registry}/vegamaps-frontend-service:${var.frontend_green_tag}"

    # Nginx
    nginx_tag           = ""   # REQUIRED
    nginx_image         = "${var.registry}/vegamaps-frontend-nginx:${var.nginx_tag}"
    active_frontend     = "blue"   # "blue" or "green" – controls which backend Nginx uses

    # Nginx environment variables (match your compose)
    nginx_rtmp_port              = 1935
    nginx_rtmp_stream_key        = "vegamaps_livestream"
    nginx_rtmp_publish_ip        = "127.0.0.1"
    nginx_record_path            = "/var/recordings"
    nginx_hls_path               = "/tmp/hls"
    nginx_hls_port               = 8080
    nginx_recording_endpoint_url = "http://127.0.0.1:8081"
    nginx_recording_endpoint_port = 8081
    nginx_recording_allow_cidr   = "10.0.0.0/8"
    nginx_ssl_cert_path          = "/etc/nginx/ssl/cert.pem"
    nginx_ssl_key_path           = "/etc/nginx/ssl/key.pem"

    # Host paths (adjust for your Nomad client)
    ssl_host_path        = "/opt/nomad/config/vegamaps/ssl"
    recordings_host_path = "/opt/nomad/volumes/vegamaps/recordings"
    hls_host_path        = "/opt/nomad/volumes/vegamaps/hls"
    nginx_template_path  = "/opt/nomad/config/vegamaps/nginx/nginx.prod.conf.template"
  }

  group "frontend" {
    network {
      mode = "bridge"
      # Internal ports for the two frontends (not exposed to host)
      port "frontend_blue" {
        static = 8080
        to     = 80
      }
      port "frontend_green" {
        static = 8081
        to     = 80
      }
      # Ports exposed by Nginx to the host
      port "nginx_http" {
        static = 80
        to     = 80
      }
      port "nginx_https" {
        static = 443
        to     = 443
      }
      port "nginx_rtmp" {
        static = var.nginx_rtmp_port
        to     = var.nginx_rtmp_port
      }
      port "nginx_hls" {
        static = var.nginx_hls_port
        to     = var.nginx_hls_port
      }
      port "nginx_recording" {
        static = var.nginx_recording_endpoint_port
        to     = var.nginx_recording_endpoint_port
      }
    }

    # ---------- Redis ----------
    task "redis" {
      driver = "docker"
      config {
        image = var.redis_image
        ports = ["redis"]   # internally 6379
      }
      args = ["redis-server", "--appendonly", "yes"]
      resources {
        cpu    = 200
        memory = 512
      }
      service {
        name = "vegamaps-redis"
        port = 6379
        check {
          type     = "tcp"
          interval = var.redis_health_interval
          timeout  = "5s"
        }
      }
      restart {
        attempts = 10
        interval = "5m"
        mode     = "fail"
      }
    }

    # ---------- Frontend Blue ----------
    task "frontend-blue" {
      driver = "docker"
      config {
        image = var.frontend_blue_image
        ports = ["frontend_blue"]
      }
      env {
        REDIS_HOST = "localhost"
        REDIS_PORT = "6379"
      }
      resources {
        cpu    = 500
        memory = 512
      }
      service {
        name = "vegamaps-frontend-blue"
        port = "frontend_blue"
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "10s"
        }
      }
      restart {
        attempts = 3
        interval = "5m"
        mode     = "fail"
      }
    }

    # ---------- Frontend Green ----------
    task "frontend-green" {
      driver = "docker"
      config {
        image = var.frontend_green_image
        ports = ["frontend_green"]
      }
      env {
        REDIS_HOST = "localhost"
        REDIS_PORT = "6379"
      }
      resources {
        cpu    = 500
        memory = 512
      }
      service {
        name = "vegamaps-frontend-green"
        port = "frontend_green"
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "10s"
        }
      }
      restart {
        attempts = 3
        interval = "5m"
        mode     = "fail"
      }
    }

    # ---------- Nginx ----------
    task "nginx" {
      driver = "docker"
      config {
        image = var.nginx_image
        ports = ["nginx_http", "nginx_https", "nginx_rtmp", "nginx_hls", "nginx_recording"]
        volumes = [
          "${var.ssl_host_path}:/etc/nginx/ssl:ro",
          "${var.recordings_host_path}:/var/recordings",
          "${var.hls_host_path}:/tmp/hls",
          "${var.nginx_template_path}:/etc/nginx/nginx.conf.template:ro"
        ]
      }
      env {
        ACTIVE_FRONTEND                = var.active_frontend == "blue" ? "vegamaps-frontend-blue" : "vegamaps-frontend-green"
        NGINX_RTMP_PORT                = var.nginx_rtmp_port
        NGINX_RTMP_STREAM_KEY          = var.nginx_rtmp_stream_key
        NGINX_RTMP_PUBLISH_IP          = var.nginx_rtmp_publish_ip
        NGINX_RECORD_PATH              = var.nginx_record_path
        NGINX_HLS_PATH                 = var.nginx_hls_path
        NGINX_HLS_PORT                 = var.nginx_hls_port
        NGINX_RECORDING_ENDPOINT_URL   = var.nginx_recording_endpoint_url
        NGINX_RECORDING_ENDPOINT_PORT  = var.nginx_recording_endpoint_port
        NGINX_RECORDING_ALLOW_CIDR     = var.nginx_recording_allow_cidr
        NGINX_SSL_CERT_PATH            = var.nginx_ssl_cert_path
        NGINX_SSL_KEY_PATH             = var.nginx_ssl_key_path
        REDIS_HOST                     = "localhost"
        REDIS_PORT                     = "6379"
      }
      command = "/bin/sh"
      args = ["-c", <<-CMD
        set -e
        if [ ! -f /etc/nginx/nginx.conf.template ]; then
          echo "ERROR: nginx template not found"; exit 1
        fi
        envsubst "$$ACTIVE_FRONTEND $$NGINX_RTMP_PORT $$NGINX_RTMP_STREAM_KEY $$NGINX_RTMP_PUBLISH_IP $$NGINX_RECORD_PATH $$NGINX_HLS_PATH $$NGINX_HLS_PORT $$NGINX_RECORDING_ENDPOINT_URL $$NGINX_RECORDING_ENDPOINT_PORT $$NGINX_RECORDING_ALLOW_CIDR $$NGINX_SSL_CERT_PATH $$NGINX_SSL_KEY_PATH $$REDIS_HOST $$REDIS_PORT" \
          < /etc/nginx/nginx.conf.template > /etc/nginx/nginx.conf
        /usr/local/nginx/sbin/nginx -t
        exec /usr/local/nginx/sbin/nginx -g "daemon off;"
      CMD
      ]
      resources {
        cpu    = 200
        memory = 256
      }
      service {
        name = "vegamaps-nginx"
        port = "nginx_http"
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "10s"
        }
      }
      restart {
        attempts = 3
        interval = "5m"
        mode     = "fail"
      }
    }
  }
}