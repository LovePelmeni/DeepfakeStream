job "observability" {
  datacenters = ["dc1"]
  type        = "service"

  # ======================== VARIABLES ========================
  variables {
    # Images & tags
    prometheus_image      = "prom/prometheus:latest"
    grafana_image         = "grafana/grafana:11.3.0"
    loki_image            = "grafana/loki:3.2.0"
    alloy_image           = "grafana/alloy:v1.15.1"
    alertmanager_image    = "prom/alertmanager:v0.27.0"
    cadvisor_image        = "gcr.io/cadvisor/cadvisor:latest"
    node_exporter_image   = "prom/node-exporter:latest"
    pyroscope_image       = "grafana/pyroscope:1.4.0"
    mysql_bugsink_image   = "mysql:8.0"
    bugsink_image         = "bugsink/bugsink:latest"

    # Ports (host ports – adjust if needed)
    prometheus_host_port  = 9090
    grafana_host_port     = 3000
    alertmanager_host_port = 9093
    cadvisor_host_port    = 8081
    node_exporter_host_port = 9100
    pyroscope_host_port   = 4040
    bugsink_host_port     = 8001

    # Alloy internal ports (exposed on host)
    alloy_http_port       = 12345
    alloy_faro_port       = 12347
    alloy_otlp_grpc_port  = 4317
    alloy_otlp_http_port  = 4318

    # Health check settings (optional, keep defaults)
    # ...

    # Grafana
    gf_security_admin_user     = "admin"          # CHANGE ME
    gf_security_admin_password = "admin"          # CHANGE ME
    gf_server_root_url         = "http://localhost:3000"
    gf_install_plugins         = "grafana-pyroscope-app,grafana-github-datasource"

    # Prometheus retention
    prometheus_retention_time = "90d"

    # Bugsink & MySQL
    mysql_bugsink_root_password = ""               # REQUIRED
    mysql_bugsink_password      = ""               # REQUIRED
    mysql_bugsink_database      = "bugsink"
    mysql_bugsink_user          = "bugsink"

    bugsink_secret_key          = ""               # REQUIRED
    bugsink_admin_user          = ""               # REQUIRED
    bugsink_admin_password      = ""               # REQUIRED
    bugsink_base_url            = "http://localhost:8001"
    bugsink_database_url        = "mysql://bugsink:${var.mysql_bugsink_password}@localhost:3306/bugsink"
    bugsink_site_title          = "Error Tracking"
    bugsink_max_events_5min     = 5000
    bugsink_max_events_hour     = 20000
    bugsink_log_level           = "info"
    bugsink_allowed_hosts       = "errors.example.com,localhost,*"
    bugsink_csrf_allowed_origins = "errors.example.com,localhost,*"

    # SMTP (optional)
    smtp_host     = ""
    smtp_user     = ""
    smtp_password = ""
    smtp_from     = "errors@example.com"
    smtp_use_tls  = "True"
    smtp_use_ssl  = "False"
    smtp_port     = "587"

    # Main network (nomad bridge network name)
    # Not used directly, tasks share group network
  }

  group "observability" {
    # Shared bridge network – all tasks share network namespace
    network {
      mode = "bridge"
      port "prometheus" {
        static = var.prometheus_host_port
        to     = 9090
      }
      port "grafana" {
        static = var.grafana_host_port
        to     = 3000
      }
      port "alertmanager" {
        static = var.alertmanager_host_port
        to     = 9093
      }
      port "cadvisor" {
        static = var.cadvisor_host_port
        to     = 8080
      }
      port "node_exporter" {
        static = var.node_exporter_host_port
        to     = 9100
      }
      port "pyroscope" {
        static = var.pyroscope_host_port
        to     = 4040
      }
      port "bugsink" {
        static = var.bugsink_host_port
        to     = 8000
      }
      # Alloy ports
      port "alloy_http" {
        static = var.alloy_http_port
        to     = 12345
      }
      port "alloy_faro" {
        static = var.alloy_faro_port
        to     = 12347
      }
      port "alloy_otlp_grpc" {
        static = var.alloy_otlp_grpc_port
        to     = 4317
      }
      port "alloy_otlp_http" {
        static = var.alloy_otlp_http_port
        to     = 4318
      }
    }

    # ---------- Prometheus ----------
    task "prometheus" {
      driver = "docker"

      config {
        image = var.prometheus_image
        ports = ["prometheus"]
        extra_hosts = ["host.docker.internal:host-gateway"]
        volumes = [
          "/opt/nomad/config/observability/prometheus.yml:/etc/prometheus/prometheus.yml:ro",
          "/opt/nomad/config/observability/alerts.yml:/etc/prometheus/rules/alerts.yml:ro",
          "/opt/nomad/volumes/observability/prometheus_data:/prometheus"
        ]
      }

      args = [
        "--config.file=/etc/prometheus/prometheus.yml",
        "--storage.tsdb.path=/prometheus",
        "--storage.tsdb.retention.time=${var.prometheus_retention_time}",
        "--web.enable-lifecycle",
        "--web.enable-admin-api"
      ]

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "prometheus"
        port = "prometheus"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
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

    # ---------- Loki ----------
    task "loki" {
      driver = "docker"

      config {
        image = var.loki_image
        volumes = [
          "/opt/nomad/config/observability/loki.yaml:/etc/loki/loki.yaml:ro",
          "/opt/nomad/volumes/observability/loki_data:/loki"
        ]
      }

      args = ["-config.file=/etc/loki/loki.yaml"]

      resources {
        cpu    = 200
        memory = 512
      }

      service {
        name = "loki"
        port = 3100
        check {
          type     = "http"
          path     = "/ready"
          interval = "10s"
          timeout  = "5s"
        }
      }
    }

    # ---------- Grafana ----------
    task "grafana" {
      driver = "docker"

      config {
        image = var.grafana_image
        ports = ["grafana"]
        volumes = [
          "/opt/nomad/config/observability/grafana/provisioning:/etc/grafana/provisioning:ro",
          "/opt/nomad/volumes/observability/grafana_data:/var/lib/grafana"
        ]
      }

      env {
        GF_SECURITY_ADMIN_USER     = var.gf_security_admin_user
        GF_SECURITY_ADMIN_PASSWORD = var.gf_security_admin_password
        GF_USERS_ALLOW_SIGN_UP     = "false"
        GF_AUTH_ANONYMOUS_ENABLED  = "false"
        GF_SERVER_ROOT_URL         = var.gf_server_root_url
        GF_INSTALL_PLUGINS         = var.gf_install_plugins
        GITHUB_ACCESS_TOKEN        = ""   # optional
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "grafana"
        port = "grafana"
        check {
          type     = "http"
          path     = "/api/health"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- Alloy ----------
    task "alloy" {
      driver = "docker"

      config {
        image = var.alloy_image
        ports = ["alloy_http", "alloy_faro", "alloy_otlp_grpc", "alloy_otlp_http"]
        volumes = [
          "/opt/nomad/config/observability/alloy.alloy:/etc/alloy/config.alloy:ro",
          "/opt/nomad/volumes/observability/alloy_data:/data"
        ]
      }

      env {
        LOG_LEVEL = "info"
      }

      args = [
        "run",
        "--server.http.listen-addr=0.0.0.0:12345",
        "--storage.path=/data",
        "/etc/alloy/config.alloy"
      ]

      resources {
        cpu    = 200
        memory = 512
      }
    }

    # ---------- Alertmanager ----------
    task "alertmanager" {
      driver = "docker"

      config {
        image = var.alertmanager_image
        ports = ["alertmanager"]
        volumes = [
          "/opt/nomad/config/observability/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro",
          "/opt/nomad/volumes/observability/alertmanager_data:/alertmanager"
        ]
      }

      args = [
        "--config.file=/etc/alertmanager/alertmanager.yml",
        "--storage.path=/alertmanager",
        "--web.listen-address=:9093"
      ]

      resources {
        cpu    = 100
        memory = 256
      }

      service {
        name = "alertmanager"
        port = "alertmanager"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- cAdvisor ----------
    task "cadvisor" {
      driver = "docker"

      config {
        image = var.cadvisor_image
        ports = ["cadvisor"]
        privileged = true
        volumes = [
          "/:/rootfs:ro",
          "/var/run:/var/run:ro",
          "/sys:/sys:ro",
          "/var/lib/docker:/var/lib/docker:ro",
          "/dev/disk:/dev/disk:ro"
        ]
      }

      resources {
        cpu    = 200
        memory = 256
      }

      service {
        name = "cadvisor"
        port = "cadvisor"
        check {
          type     = "http"
          path     = "/healthz"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- Node Exporter ----------
    task "node-exporter" {
      driver = "docker"

      config {
        image = var.node_exporter_image
        ports = ["node_exporter"]
        volumes = [
          "/proc:/host/proc:ro",
          "/sys:/host/sys:ro",
          "/:/rootfs:ro"
        ]
      }

      args = [
        "--path.procfs=/host/proc",
        "--path.sysfs=/host/sys",
        "--path.rootfs=/rootfs",
        "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
      ]

      resources {
        cpu    = 100
        memory = 128
      }

      service {
        name = "node-exporter"
        port = "node_exporter"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- Pyroscope ----------
    task "pyroscope" {
      driver = "docker"

      config {
        image = var.pyroscope_image
        ports = ["pyroscope"]
        volumes = [
          "/opt/nomad/volumes/observability/pyroscope_data:/var/lib/pyroscope"
        ]
      }

      args = ["server"]

      resources {
        cpu    = 200
        memory = 512
      }

      service {
        name = "pyroscope"
        port = "pyroscope"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- MySQL for Bugsink ----------
    task "mysql-bugsink" {
      driver = "docker"

      config {
        image = var.mysql_bugsink_image
        volumes = [
          "/opt/nomad/volumes/observability/mysql_bugsink_data:/var/lib/mysql"
        ]
      }

      env {
        MYSQL_ROOT_PASSWORD = var.mysql_bugsink_root_password
        MYSQL_DATABASE      = var.mysql_bugsink_database
        MYSQL_USER          = var.mysql_bugsink_user
        MYSQL_PASSWORD      = var.mysql_bugsink_password
      }

      resources {
        cpu    = 200
        memory = 512
      }

      service {
        name = "mysql-bugsink"
        port = 3306
        check {
          type     = "tcp"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }

    # ---------- Bugsink ----------
    task "bugsink" {
      driver = "docker"

      config {
        image = var.bugsink_image
        ports = ["bugsink"]
        volumes = [
          "/opt/nomad/volumes/observability/bugsink_data:/data"
        ]
      }

      env {
        SECRET_KEY                         = var.bugsink_secret_key
        CREATE_SUPERUSER                   = "${var.bugsink_admin_user}:${var.bugsink_admin_password}"
        PORT                               = "8000"
        BASE_URL                           = var.bugsink_base_url
        SITE_TITLE                         = var.bugsink_site_title
        DATABASE_URL                       = var.bugsink_database_url
        BEHIND_HTTPS_PROXY                 = "True"
        MAX_EVENTS_PER_PROJECT_PER_5_MINUTES = var.bugsink_max_events_5min
        MAX_EVENTS_PER_PROJECT_PER_HOUR    = var.bugsink_max_events_hour
        EMAIL_HOST                         = var.smtp_host
        EMAIL_HOST_USER                    = var.smtp_user
        EMAIL_HOST_PASSWORD                = var.smtp_password
        DEFAULT_FROM_EMAIL                 = var.smtp_from
        EMAIL_USE_TLS                      = var.smtp_use_tls
        EMAIL_USE_SSL                      = var.smtp_use_ssl
        EMAIL_PORT                         = var.smtp_port
        LOG_LEVEL                          = var.bugsink_log_level
        ALLOWED_HOSTS                      = var.bugsink_allowed_hosts
        CSRF_ALLOWED_ORIGINS               = var.bugsink_csrf_allowed_origins
        SECURE_PROXY_SSL_HEADER            = "(\"HTTP_X_FORWARDED_PROTO\", \"https\")"
        SESSION_COOKIE_SECURE              = "true"
        CSRF_COOKIE_SECURE                 = "true"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "bugsink"
        port = "bugsink"
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