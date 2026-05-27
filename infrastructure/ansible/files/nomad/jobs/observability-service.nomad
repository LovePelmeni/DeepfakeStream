# ==================================================================
# Nomad job for full observability stack (production)
# ==================================================================

job "observability" {
  region      = "global"
  datacenters = ["dc1"]
  type        = "service"

  # ------------------------------------------------------------------
  # Prometheus
  # ------------------------------------------------------------------
  group "prometheus" {
    count = 1
    network {
      port "http" {
        to = 9090
        static = var.prometheus_port
      }
    }
    volume "prometheus_data" {
      type      = "host"
      source    = "prometheus_data"
      read_only = false
    }
    volume "prometheus_config" {
      type      = "host"
      source    = "prometheus_config"
      read_only = true
    }
    volume "prometheus_rules" {
      type      = "host"
      source    = "prometheus_rules"
      read_only = true
    }
    task "prometheus" {
      driver = "docker"
      config {
        image = var.prometheus_image
        ports = ["http"]
        args = [
          "--config.file=/etc/prometheus/prometheus.yml",
          "--storage.tsdb.path=/prometheus",
          "--storage.tsdb.retention.time=${var.prometheus_retention_time}",
          "--web.enable-lifecycle",
          "--web.enable-admin-api"
        ]
      }
      volume_mount {
        volume      = "prometheus_config"
        destination = "/etc/prometheus/prometheus.yml"
        read_only   = true
      }
      volume_mount {
        volume      = "prometheus_rules"
        destination = "/etc/prometheus/rules/alerts.yml"
        read_only   = true
      }
      volume_mount {
        volume      = "prometheus_data"
        destination = "/prometheus"
      }
      resources {
        cpu    = var.prometheus_cpu
        memory = var.prometheus_memory
      }
      service {
        name = "prometheus"
        port = "http"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = var.prometheus_health_interval
          timeout  = var.prometheus_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Loki
  # ------------------------------------------------------------------
  group "loki" {
    count = 1
    network {
      port "http" {
        to = 3100
        static = var.loki_port
      }
    }
    volume "loki_data" {
      type      = "host"
      source    = "loki_data"
      read_only = false
    }
    volume "loki_config" {
      type      = "host"
      source    = "loki_config"
      read_only = true
    }
    task "loki" {
      driver = "docker"
      config {
        image = var.loki_image
        ports = ["http"]
        args = ["-config.file=/etc/loki/loki.yaml"]
      }
      volume_mount {
        volume      = "loki_config"
        destination = "/etc/loki/loki.yaml"
        read_only   = true
      }
      volume_mount {
        volume      = "loki_data"
        destination = "/loki"
      }
      resources {
        cpu    = var.loki_cpu
        memory = var.loki_memory
      }
      service {
        name = "loki"
        port = "http"
        check {
          type     = "http"
          path     = "/ready"
          interval = var.loki_health_interval
          timeout  = var.loki_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Alloy (metrics & logs collector)
  # ------------------------------------------------------------------
  group "alloy" {
    count = 1
    network {
      port "http"      { to = 12345; static = var.alloy_http_port }
      port "faro"      { to = 12347; static = var.alloy_faro_port }
      port "otlp_grpc" { to = 4317;  static = var.alloy_otlp_grpc_port }
      port "otlp_http" { to = 4318;  static = var.alloy_otlp_http_port }
    }
    volume "alloy_data" {
      type      = "host"
      source    = "alloy_data"
      read_only = false
    }
    volume "alloy_config" {
      type      = "host"
      source    = "alloy_config"
      read_only = true
    }
    task "alloy" {
      driver = "docker"
      config {
        image = var.alloy_image
        ports = ["http", "faro", "otlp_grpc", "otlp_http"]
        args = [
          "run",
          "--server.http.listen-addr=0.0.0.0:12345",
          "--storage.path=/data",
          "/etc/alloy/config.alloy"
        ]
      }
      volume_mount {
        volume      = "alloy_config"
        destination = "/etc/alloy/config.alloy"
        read_only   = true
      }
      volume_mount {
        volume      = "alloy_data"
        destination = "/data"
      }
      env {
        LOG_LEVEL = var.alloy_log_level
      }
      resources {
        cpu    = var.alloy_cpu
        memory = var.alloy_memory
      }
      service {
        name = "alloy"
        port = "http"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Tempo (traces)
  # ------------------------------------------------------------------
  group "tempo" {
    count = 1
    network {
      port "http"  { to = 3200; static = var.tempo_port }
      port "query" { to = 9095; static = var.tempo_query_port }
    }
    volume "tempo_data" {
      type      = "host"
      source    = "tempo_data"
      read_only = false
    }
    volume "tempo_config" {
      type      = "host"
      source    = "tempo_config"
      read_only = true
    }
    task "tempo" {
      driver = "docker"
      config {
        image = var.tempo_image
        ports = ["http", "query"]
        args = ["--config.file=/etc/tempo/tempo-config.yaml"]
      }
      volume_mount {
        volume      = "tempo_config"
        destination = "/etc/tempo/tempo-config.yaml"
        read_only   = true
      }
      volume_mount {
        volume      = "tempo_data"
        destination = "/var/tempo/traces"
      }
      resources {
        cpu    = var.tempo_cpu
        memory = var.tempo_memory
      }
      service {
        name = "tempo"
        port = "http"
        check {
          type     = "http"
          path     = "/ready"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Alertmanager
  # ------------------------------------------------------------------
  group "alertmanager" {
    count = 1
    network {
      port "http" {
        to = 9093
        static = var.alertmanager_port
      }
    }
    volume "alertmanager_data" {
      type      = "host"
      source    = "alertmanager_data"
      read_only = false
    }
    volume "alertmanager_config" {
      type      = "host"
      source    = "alertmanager_config"
      read_only = true
    }
    task "alertmanager" {
      driver = "docker"
      config {
        image = var.alertmanager_image
        ports = ["http"]
        args = [
          "--config.file=/etc/alertmanager/alertmanager.yml",
          "--storage.path=/alertmanager",
          "--web.listen-address=:9093"
        ]
      }
      volume_mount {
        volume      = "alertmanager_config"
        destination = "/etc/alertmanager/alertmanager.yml"
        read_only   = true
      }
      volume_mount {
        volume      = "alertmanager_data"
        destination = "/alertmanager"
      }
      resources {
        cpu    = var.alertmanager_cpu
        memory = var.alertmanager_memory
      }
      service {
        name = "alertmanager"
        port = "http"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # cAdvisor (requires host mounts & privileged)
  # ------------------------------------------------------------------
  group "cadvisor" {
    count = 1
    network {
      port "http" {
        to = 8080
        static = var.cadvisor_port
      }
    }
    task "cadvisor" {
      driver = "docker"
      privileged = true
      config {
        image = var.cadvisor_image
        ports = ["http"]
        volumes = [
          "/:/rootfs:ro",
          "/var/run:/var/run:ro",
          "/sys:/sys:ro",
          "/var/lib/docker:/var/lib/docker:ro",
          "/dev/disk:/dev/disk:ro"
        ]
      }
      resources {
        cpu    = var.cadvisor_cpu
        memory = var.cadvisor_memory
      }
      service {
        name = "cadvisor"
        port = "http"
        check {
          type     = "http"
          path     = "/healthz"
          interval = var.cadvisor_health_interval
          timeout  = var.cadvisor_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Node Exporter (host metrics)
  # ------------------------------------------------------------------
  group "node-exporter" {
    count = 1
    network {
      port "http" {
        to = 9100
        static = var.node_exporter_port
      }
    }
    task "node-exporter" {
      driver = "docker"
      config {
        image = var.node_exporter_image
        ports = ["http"]
        volumes = [
          "/proc:/host/proc:ro",
          "/sys:/host/sys:ro",
          "/:/rootfs:ro"
        ]
        args = [
          "--path.procfs=/host/proc",
          "--path.sysfs=/host/sys",
          "--path.rootfs=/rootfs",
          "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
        ]
      }
      resources {
        cpu    = var.node_exporter_cpu
        memory = var.node_exporter_memory
      }
      service {
        name = "node-exporter"
        port = "http"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = var.node_exporter_health_interval
          timeout  = var.node_exporter_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Pyroscope (continuous profiling)
  # ------------------------------------------------------------------
  group "pyroscope" {
    count = 1
    network {
      port "http" {
        to = 4040
        static = var.pyroscope_port
      }
    }
    volume "pyroscope_data" {
      type      = "host"
      source    = "pyroscope_data"
      read_only = false
    }
    task "pyroscope" {
      driver = "docker"
      config {
        image = var.pyroscope_image
        ports = ["http"]
        args = [var.pyroscope_command]
      }
      volume_mount {
        volume      = "pyroscope_data"
        destination = "/var/lib/pyroscope"
      }
      resources {
        cpu    = var.pyroscope_cpu
        memory = var.pyroscope_memory
      }
      service {
        name = "pyroscope"
        port = "http"
        check {
          type     = "http"
          path     = "/-/healthy"
          interval = var.pyroscope_health_interval
          timeout  = var.pyroscope_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # MySQL for Bugsink (stateful)
  # ------------------------------------------------------------------
  group "mysql-bugsink" {
    count = 1
    network {
      port "mysql" {
        to = 3306
        static = var.mysql_bugsink_port
      }
    }
    volume "mysql_bugsink_data" {
      type      = "host"
      source    = "mysql_bugsink_data"
      read_only = false
    }
    task "mysql" {
      driver = "docker"
      config {
        image = var.mysql_bugsink_image
        ports = ["mysql"]
      }
      env {
        MYSQL_ROOT_PASSWORD = var.mysql_bugsink_root_password
        MYSQL_DATABASE      = var.mysql_bugsink_database
        MYSQL_USER          = var.mysql_bugsink_user
        MYSQL_PASSWORD      = var.mysql_bugsink_password
      }
      volume_mount {
        volume      = "mysql_bugsink_data"
        destination = "/var/lib/mysql"
      }
      resources {
        cpu    = var.mysql_bugsink_cpu
        memory = var.mysql_bugsink_memory
      }
      service {
        name = "mysql-bugsink"
        port = "mysql"
        check {
          type     = "script"
          command  = "mysqladmin"
          args     = ["ping", "-h", "localhost"]
          interval = var.mysql_bugsink_health_interval
          timeout  = var.mysql_bugsink_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Bugsink (error tracking)
  # ------------------------------------------------------------------
  group "bugsink" {
    count = 1
    network {
      port "http" {
        to = 8000
        static = var.bugsink_port
      }
    }
    volume "bugsink_data" {
      type      = "host"
      source    = "bugsink_data"
      read_only = false
    }
    task "bugsink" {
      driver = "docker"
      config {
        image = var.bugsink_image
        ports = ["http"]
      }
      env {
        SECRET_KEY                      = var.bugsink_secret_key
        CREATE_SUPERUSER                = "${var.bugsink_admin_user}:${var.bugsink_admin_password}"
        PORT                            = "8000"
        BASE_URL                        = var.bugsink_base_url
        SITE_TITLE                      = var.bugsink_site_title
        DATABASE_URL                    = var.bugsink_database_url
        BEHIND_HTTPS_PROXY              = var.bugsink_behind_https_proxy
        MAX_EVENTS_PER_PROJECT_PER_5_MINUTES = var.bugsink_max_events_5min
        MAX_EVENTS_PER_PROJECT_PER_HOUR      = var.bugsink_max_events_hour
        EMAIL_HOST                      = var.smtp_host
        EMAIL_HOST_USER                 = var.smtp_user
        EMAIL_HOST_PASSWORD             = var.smtp_password
        DEFAULT_FROM_EMAIL              = var.smtp_from_email
        EMAIL_USE_TLS                   = var.smtp_use_tls
        EMAIL_USE_SSL                   = var.smtp_use_ssl
        EMAIL_PORT                      = var.smtp_port
        LOG_LEVEL                       = var.bugsink_log_level
        ALLOWED_HOSTS                   = var.bugsink_allowed_hosts
        CSRF_ALLOWED_ORIGINS            = var.bugsink_csrf_allowed_origins
        SECURE_PROXY_SSL_HEADER         = "(\"HTTP_X_FORWARDED_PROTO\", \"https\")"
        SESSION_COOKIE_SECURE            = "true"
        CSRF_COOKIE_SECURE               = "true"
      }
      volume_mount {
        volume      = "bugsink_data"
        destination = "/data"
      }
      resources {
        cpu    = var.bugsink_cpu
        memory = var.bugsink_memory
      }
      service {
        name = "bugsink"
        port = "http"
        check {
          type     = "http"
          path     = "/"
          interval = var.bugsink_health_interval
          timeout  = var.bugsink_health_timeout
        }
      }
    }
  }

  # ------------------------------------------------------------------
  # Grafana (depends on prometheus, loki, alloy - but Nomad handles ordering via health checks)
  # ------------------------------------------------------------------
  group "grafana" {
    count = 1
    network {
      port "http" {
        to = 3000
        static = var.grafana_port
      }
    }
    volume "grafana_data" {
      type      = "host"
      source    = "grafana_data"
      read_only = false
    }
    volume "grafana_provisioning" {
      type      = "host"
      source    = "grafana_provisioning"
      read_only = true
    }
    task "grafana" {
      driver = "docker"
      config {
        image = var.grafana_image
        ports = ["http"]
      }
      env {
        GF_SECURITY_ADMIN_USER     = var.grafana_admin_user
        GF_SECURITY_ADMIN_PASSWORD = var.grafana_admin_password
        GF_USERS_ALLOW_SIGN_UP     = var.grafana_allow_sign_up
        GF_AUTH_ANONYMOUS_ENABLED  = var.grafana_anonymous_enabled
        GF_SERVER_ROOT_URL         = var.grafana_root_url
        GF_INSTALL_PLUGINS         = var.grafana_install_plugins
        GITHUB_ACCESS_TOKEN        = var.github_access_token
      }
      volume_mount {
        volume      = "grafana_provisioning"
        destination = "/etc/grafana/provisioning"
        read_only   = true
      }
      volume_mount {
        volume      = "grafana_data"
        destination = "/var/lib/grafana"
      }
      resources {
        cpu    = var.grafana_cpu
        memory = var.grafana_memory
      }
      service {
        name = "grafana"
        port = "http"
        check {
          type     = "http"
          path     = "/api/health"
          interval = "30s"
          timeout  = "5s"
        }
      }
    }
  }
}