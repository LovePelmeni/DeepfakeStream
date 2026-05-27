# jobs/frontend-stack.nomad
# ==================================================================
# Nomad job: Redis + Frontend (single instance) + Nginx router
# ==================================================================

job "frontend-stack" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"

  # Rolling update for the whole job (except Redis – stateful)
  update {
    max_parallel     = 1
    health_check     = "task_states"
    min_healthy_time = "30s"
    auto_revert      = true
  }

  # ========================== REDIS (Stateful) ==========================
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
      source    = "vegamaps_redis_data"
      read_only = false
    }

    task "redis" {
      driver = "docker"

      config {
        image = "${REDIS_IMAGE:-redis:7-alpine}"
        ports = ["redis"]
        args = ["redis-server", "--appendonly", "yes"]
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
          interval = "10s"
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
        max_file_size = 10
      }
    }
  }

  # ========================== FRONTEND (Single) ==========================
  group "frontend" {
    count = 1

    network {
      mode = "bridge"
      port "http" {
        to = 80
      }
    }

    task "frontend" {
      driver = "docker"

      config {
        image = "cr.yandex/${YANDEX_REGISTRY_ID}/vegamaps-frontend-service:${FRONTEND_TAG}"
        ports = ["http"]
      }

      env {
        REDIS_HOST = "redis.service.consul"
        REDIS_PORT = "6379"
      }

      resources {
        cpu    = 200
        memory = 256
      }

      service {
        name = "frontend"
        port = "http"
        tags = ["frontend", "production"]
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "10s"
        }
      }

      logs {
        max_files     = 3
        max_file_size = 10
      }
    }
  }

  # ========================== NGINX (Router) ==========================
  group "nginx" {
    count = 1

    network {
      mode = "bridge"
      port "http"   { to = 80;   static = 80   }
      port "https"  { to = 443;  static = 443  }
      port "rtmp"   { to = 1935; static = 1935 }
      port "hls"    { to = 8080; static = 8080 }
      port "record" { to = 8081; static = 8081 }
    }

    # Host volumes for SSL, recordings, HLS
    volume "ssl" {
      type      = "host"
      source    = "vegamaps_ssl"
      read_only = true
    }
    volume "recordings" {
      type      = "host"
      source    = "vegamaps_recordings"
      read_only = false
    }
    volume "hls" {
      type      = "host"
      source    = "vegamaps_hls"
      read_only = false
    }

    task "nginx" {
      driver = "docker"

      config {
        image = "cr.yandex/${YANDEX_REGISTRY_ID}/vegamaps-frontend-nginx:${NGINX_TAG}"
        ports = ["http", "https", "rtmp", "hls", "record"]
        volumes = [
          "/local/nginx.conf:/etc/nginx/nginx.conf:ro"
        ]
      }

      volume_mount {
        volume      = "ssl"
        destination = "/etc/nginx/ssl"
        read_only   = true
      }
      volume_mount {
        volume      = "recordings"
        destination = "/var/recordings"
      }
      volume_mount {
        volume      = "hls"
        destination = "/tmp/hls"
      }

      # ######################### NGINX CONFIGURATION TEMPLATE #########################
      template {
        data = <<EOF
# nginx.conf.template – rendered by envsubst at container start
worker_processes auto;
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        # Use Consul DNS – queries "frontend.service.consul"
        server frontend.service.consul:80 max_fails=3 fail_timeout=30s;
    }

    server {
        listen 80;
        server_name _;
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }

    server {
        listen 443 ssl;
        server_name _;
        ssl_certificate     ${NGINX_SSL_CERT_PATH};
        ssl_certificate_key ${NGINX_SSL_KEY_PATH};
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
        }
    }
}

# RTMP module configuration (if available)
rtmp {
    server {
        listen ${NGINX_RTMP_PORT};
        application live {
            live on;
            record all;
            record_path ${NGINX_RECORD_PATH};
            record_suffix all.flv;
            hls on;
            hls_path ${NGINX_HLS_PATH};
            hls_fragment 5s;
            push rtmp://${NGINX_RTMP_PUBLISH_IP}:${NGINX_RTMP_PORT}/live/${NGINX_RTMP_STREAM_KEY};
            exec_record_done /usr/local/bin/notify.sh ${NGINX_RECORDING_ENDPOINT_URL} $path $basename;
        }
    }
}

server {
    listen ${NGINX_HLS_PORT};
    location /hls {
        types {
            application/vnd.apple.mpegurl m3u8;
            video/mp2t ts;
        }
        alias ${NGINX_HLS_PATH};
        add_header Cache-Control no-cache;
    }
}

server {
    listen ${NGINX_RECORDING_ENDPOINT_PORT};
    allow ${NGINX_RECORDING_ALLOW_CIDR};
    deny all;
    location / {
        return 200 "OK";
    }
}
EOF
        destination = "/local/nginx.conf.template"
        change_mode = "signal"
        change_signal = "SIGHUP"
      }

      env {
        NGINX_RTMP_PORT              = "${NGINX_RTMP_PORT:-1935}"
        NGINX_RTMP_STREAM_KEY        = "${NGINX_RTMP_STREAM_KEY:-vegamaps_livestream}"
        NGINX_RTMP_PUBLISH_IP        = "${NGINX_RTMP_PUBLISH_IP:-127.0.0.1}"
        NGINX_RECORD_PATH            = "${NGINX_RECORD_PATH:-/var/recordings}"
        NGINX_HLS_PATH               = "${NGINX_HLS_PATH:-/tmp/hls}"
        NGINX_HLS_PORT               = "${NGINX_HLS_PORT:-8080}"
        NGINX_RECORDING_ENDPOINT_URL = "${NGINX_RECORDING_ENDPOINT_URL:-http://127.0.0.1:8081}"
        NGINX_RECORDING_ENDPOINT_PORT= "${NGINX_RECORDING_ENDPOINT_PORT:-8081}"
        NGINX_RECORDING_ALLOW_CIDR   = "${NGINX_RECORDING_ALLOW_CIDR:-10.0.0.0/8}"
        NGINX_SSL_CERT_PATH          = "${NGINX_SSL_CERT_PATH:-/etc/nginx/ssl/cert.pem}"
        NGINX_SSL_KEY_PATH           = "${NGINX_SSL_KEY_PATH:-/etc/nginx/ssl/key.pem}"
      }

      config {
        command = "sh"
        args = [
          "-c",
          "envsubst '$$NGINX_RTMP_PORT $$NGINX_RTMP_STREAM_KEY $$NGINX_RTMP_PUBLISH_IP $$NGINX_RECORD_PATH $$NGINX_HLS_PATH $$NGINX_HLS_PORT $$NGINX_RECORDING_ENDPOINT_URL $$NGINX_RECORDING_ENDPOINT_PORT $$NGINX_RECORDING_ALLOW_CIDR $$NGINX_SSL_CERT_PATH $$NGINX_SSL_KEY_PATH' < /local/nginx.conf.template > /etc/nginx/nginx.conf && /usr/local/nginx/sbin/nginx -t && exec /usr/local/nginx/sbin/nginx -g 'daemon off;'"
        ]
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      service {
        name = "nginx"
        port = "http"
        check {
          type     = "http"
          path     = "/health"
          interval = "30s"
          timeout  = "10s"
        }
      }

      logs {
        max_files     = 3
        max_file_size = 10
      }
    }
  }
}