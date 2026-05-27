job "misp" {
  datacenters = ["dc1"]
  type        = "service"

  # Variables – define defaults or pass via -var / var-file
  variables {
    # Core
    core_running_tag      = "latest"
    modules_running_tag   = "latest"
    guard_running_tag     = "latest"
    base_url              = "https://localhost"
    admin_email           = "admin@example.com"
    misp_contact          = "contact@example.com"
    misp_email            = "misp@example.com"
    admin_password        = "ChangeMe123!"
    admin_key             = ""
    admin_org             = "ORGNAME"
    admin_org_uuid        = ""
    gpg_passphrase        = ""
    encryption_key        = ""
    salt                  = ""
    uuid                  = ""
    # Database
    mysql_user            = "misp"
    mysql_password        = "example"
    mysql_root_password   = "password"
    mysql_database        = "misp"
    innodb_buffer_pool_size = "2048M"
    # Redis
    redis_password        = "redispassword"
    enable_redis_empty_password = "false"
    disable_redis_snapshot = "false"
    # SMTP (mail relay)
    smarthost_address     = ""
    smarthost_port        = "587"
    smarthost_user        = ""
    smarthost_password    = ""
    smarthost_aliases     = ""
    # Optional guard
    enable_guard          = "false"
    guard_port            = "8888"
    # Misc
    tz                    = "UTC"
    debug                 = "false"
  }

  group "misp" {
    # Shared network – all tasks in this group can use localhost
    network {
      mode = "bridge"
      port "http" {
        static = 80
        to     = 80
      }
      port "https" {
        static = 443
        to     = 443
      }
      port "guard" {
        static = var.guard_port
        to     = var.guard_port
      }
    }

    # ---------- SMTP relay (mail) ----------
    task "mail" {
      driver = "docker"
      config {
        image = "ghcr.io/egos-tech/smtp:1.1.3"
      }
      env {
        SES_USER            = "${var.ses_user}"
        SES_PASSWORD        = "${var.ses_password}"
        SES_REGION          = "${var.ses_region}"
        SES_PORT            = "${var.ses_port}"
        SMARTHOST_ADDRESS   = "${var.smarthost_address}"
        SMARTHOST_PORT      = "${var.smarthost_port}"
        SMARTHOST_USER      = "${var.smarthost_user}"
        SMARTHOST_PASSWORD  = "${var.smarthost_password}"
        SMARTHOST_ALIASES   = "${var.smarthost_aliases}"
        TZ                  = "${var.tz}"
      }
      resources {
        cpu    = 100
        memory = 128
      }
    }

    # ---------- Redis (valkey) ----------
    task "redis" {
      driver = "docker"
      config {
        image = "valkey/valkey:7.2"
        command = "/bin/sh"
        args = [
          "-c", <<-CMD
            if [ "$${DISABLE_REDIS_SNAPSHOT}" = "true" ]; then
                redis_snapshot="--save \"\""
            else
                redis_snapshot=""
            fi
            if [ "$${ENABLE_REDIS_EMPTY_PASSWORD:-false}" = "true" ]; then
              exec valkey-server $${redis_snapshot}
            else
              exec valkey-server --requirepass "$${REDIS_PASSWORD:-redispassword}" $${redis_snapshot}
            fi
          CMD
        ]
      }
      env {
        ENABLE_REDIS_EMPTY_PASSWORD = "${var.enable_redis_empty_password}"
        REDIS_PASSWORD              = "${var.redis_password}"
        TZ                          = "${var.tz}"
        DISABLE_REDIS_SNAPSHOT      = "${var.disable_redis_snapshot}"
        REDIS_PORT                  = "6379"
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
          interval = "2s"
          timeout  = "1s"
          check_restart {
            limit = 3
            grace = "5s"
          }
        }
      }
    }

    # ---------- MariaDB ----------
    task "db" {
      driver = "docker"
      config {
        image = "mariadb:10.11"
        command = "mysqld"
        args = [
          "--innodb-buffer-pool-size=${var.innodb_buffer_pool_size}",
          "--innodb-change-buffering=${var.innodb_change_buffering:-none}",
          "--innodb-io-capacity=${var.innodb_io_capacity:-1000}",
          "--innodb-io-capacity-max=${var.innodb_io_capacity_max:-2000}",
          "--innodb-log-file-size=${var.innodb_log_file_size:-600M}",
          "--innodb-read-io-threads=${var.innodb_read_io_threads:-16}",
          "--innodb-stats-persistent=${var.innodb_stats_persistent:-ON}",
          "--innodb-write-io-threads=${var.innodb_write_io_threads:-4}"
        ]
        volumes = [
          "/opt/nomad/volumes/misp/mysql_data:/var/lib/mysql"
        ]
        cap_add = ["SYS_NICE"]
      }
      env {
        MYSQL_USER          = "${var.mysql_user}"
        MYSQL_PASSWORD      = "${var.mysql_password}"
        MYSQL_ROOT_PASSWORD = "${var.mysql_root_password}"
        MYSQL_DATABASE      = "${var.mysql_database}"
        TZ                  = "${var.tz}"
      }
      resources {
        cpu    = 1000
        memory = 2048
      }
      service {
        name = "mysql"
        port = 3306
        check {
          type     = "tcp"
          interval = "2s"
          timeout  = "1s"
          check_restart {
            limit = 3
            grace = "30s"
          }
        }
      }
    }

    # ---------- MISP modules ----------
    task "misp-modules" {
      driver = "docker"
      config {
        image = "ghcr.io/misp/misp-docker/misp-modules:${var.modules_running_tag}"
        volumes = [
          "/opt/nomad/config/misp/custom/action_mod:/custom/action_mod",
          "/opt/nomad/config/misp/custom/expansion:/custom/expansion",
          "/opt/nomad/config/misp/custom/export_mod:/custom/export_mod",
          "/opt/nomad/config/misp/custom/import_mod:/custom/import_mod"
        ]
      }
      env {
        TZ = "${var.tz}"
      }
      resources {
        cpu    = 200
        memory = 512
      }
      service {
        name = "misp-modules"
        port = 6666
        check {
          type     = "tcp"
          interval = "2s"
          timeout  = "1s"
          check_restart {
            limit = 3
            grace = "5s"
          }
        }
      }
    }

    # ---------- MISP core ----------
    task "misp-core" {
      driver = "docker"
      config {
        image = "ghcr.io/misp/misp-docker/misp-core:${var.core_running_tag}"
        ports = ["http", "https"]
        volumes = [
          "/opt/nomad/config/misp/configs:/var/www/MISP/app/Config",
          "/opt/nomad/config/misp/logs:/var/www/MISP/app/tmp/logs",
          "/opt/nomad/config/misp/files:/var/www/MISP/app/files",
          "/opt/nomad/config/misp/ssl:/etc/nginx/certs",
          "/opt/nomad/config/misp/gnupg:/var/www/MISP/.gnupg",
          "/opt/nomad/volumes/misp/misp_guard_ca:/usr/local/share/ca-certificates/misp_guard"
        ]
        cap_add = ["AUDIT_WRITE"]
      }
      env {
        # Core settings
        BASE_URL                           = "${var.base_url}"
        CRON_USER_ID                       = "${var.cron_user_id}"
        CRON_PULLALL                       = "${var.cron_pullall}"
        CRON_PUSHALL                       = "${var.cron_pushall}"
        DISABLE_IPV6                       = "${var.disable_ipv6}"
        DISABLE_SSL_REDIRECT               = "${var.disable_ssl_redirect}"
        ENABLE_DB_SETTINGS                 = "${var.enable_db_settings}"
        ENABLE_BACKGROUND_UPDATES          = "${var.enable_background_updates}"
        ENCRYPTION_KEY                     = "${var.encryption_key}"
        SALT                               = "${var.salt}"
        UUID                               = "${var.uuid}"
        DISABLE_CA_REFRESH                 = "${var.disable_ca_refresh}"
        DISABLE_PRINTING_PLAINTEXT_CREDENTIALS = "${var.disable_printing_plaintext_credentials}"
        # Admin / org
        ADMIN_EMAIL                        = "${var.admin_email}"
        MISP_CONTACT                       = "${var.misp_contact}"
        MISP_EMAIL                         = "${var.misp_email}"
        ADMIN_PASSWORD                     = "${var.admin_password}"
        ADMIN_KEY                          = "${var.admin_key}"
        ADMIN_ORG                          = "${var.admin_org}"
        ADMIN_ORG_UUID                     = "${var.admin_org_uuid}"
        GPG_PASSPHRASE                     = "${var.gpg_passphrase}"
        ATTACHMENTS_DIR                    = "${var.attachments_dir}"
        TZ                                 = "${var.tz}"
        ENABLE_THEMES                      = "${var.enable_themes}"
        # Database
        MYSQL_HOST                         = "localhost"
        MYSQL_PORT                         = "3306"
        MYSQL_USER                         = "${var.mysql_user}"
        MYSQL_PASSWORD                     = "${var.mysql_password}"
        MYSQL_DATABASE                     = "${var.mysql_database}"
        MYSQL_TLS                          = "${var.mysql_tls}"
        MYSQL_TLS_CA                       = "${var.mysql_tls_ca}"
        MYSQL_TLS_CERT                     = "${var.mysql_tls_cert}"
        MYSQL_TLS_KEY                      = "${var.mysql_tls_key}"
        # Redis
        REDIS_HOST                         = "localhost"
        REDIS_PORT                         = "6379"
        REDIS_PASSWORD                     = "${var.redis_password}"
        ENABLE_REDIS_EMPTY_PASSWORD        = "${var.enable_redis_empty_password}"
        # Proxy
        PROXY_ENABLE                       = "${var.proxy_enable}"
        PROXY_HOST                         = "${var.proxy_host}"
        PROXY_PORT                         = "${var.proxy_port}"
        PROXY_METHOD                       = "${var.proxy_method}"
        PROXY_USER                         = "${var.proxy_user}"
        PROXY_PASSWORD                     = "${var.proxy_password}"
        # S3
        S3_BUCKET                          = "${var.s3_bucket}"
        S3_ENDPOINT                        = "${var.s3_endpoint}"
        S3_ACCESS_KEY                      = "${var.s3_access_key}"
        S3_SECRET_KEY                      = "${var.s3_secret_key}"
        # Stunnel
        STUNNEL                            = "${var.stunnel}"
        STUNNEL_CONFIG                     = "${var.stunnel_config}"
        # Supervisor
        SUPERVISOR_HOST                    = "${var.supervisor_host}"
        SUPERVISOR_USERNAME                = "${var.supervisor_username}"
        SUPERVISOR_PASSWORD                = "${var.supervisor_password}"
        # Sync servers
        SYNCSERVERS                        = "${var.syncservers}"
        SYNCSERVERS_1_DATA                 = "${var.syncservers_1_data}"
        # PHP / nginx
        FASTCGI_READ_TIMEOUT               = "${var.fastcgi_read_timeout}"
        FASTCGI_SEND_TIMEOUT               = "${var.fastcgi_send_timeout}"
        FASTCGI_CONNECT_TIMEOUT            = "${var.fastcgi_connect_timeout}"
        FASTCGI_STATUS_LISTEN              = "${var.fastcgi_status_listen}"
        PHP_MEMORY_LIMIT                   = "${var.php_memory_limit}"
        PHP_MAX_EXECUTION_TIME             = "${var.php_max_execution_time}"
        PHP_UPLOAD_MAX_FILESIZE            = "${var.php_upload_max_filesize}"
        PHP_POST_MAX_SIZE                  = "${var.php_post_max_size}"
        PHP_MAX_INPUT_TIME                 = "${var.php_max_input_time}"
        PHP_MAX_FILE_UPLOADS               = "${var.php_max_file_uploads}"
        PHP_FCGI_CHILDREN                  = "${var.php_fcgi_children}"
        PHP_FCGI_START_SERVERS             = "${var.php_fcgi_start_servers}"
        PHP_FCGI_SPARE_SERVERS             = "${var.php_fcgi_spare_servers}"
        PHP_FCGI_MAX_REQUESTS              = "${var.php_fcgi_max_requests}"
        PHP_SESSION_TIMEOUT                = "${var.php_session_timeout}"
        PHP_SESSION_COOKIE_TIMEOUT         = "${var.php_session_cookie_timeout}"
        PHP_SESSION_DEFAULTS               = "${var.php_session_defaults}"
        PHP_SESSION_AUTO_REGENERATE        = "${var.php_session_auto_regenerate}"
        PHP_SESSION_CHECK_AGENT            = "${var.php_session_check_agent}"
        PHP_SESSION_COOKIE_SECURE          = "${var.php_session_cookie_secure}"
        PHP_SESSION_COOKIE_DOMAIN          = "${var.php_session_cookie_domain}"
        PHP_SESSION_COOKIE_SAMESITE        = "${var.php_session_cookie_samesite}"
        # Security headers
        HSTS_MAX_AGE                       = "${var.hsts_max_age}"
        X_FRAME_OPTIONS                    = "${var.x_frame_options}"
        CONTENT_SECURITY_POLICY            = "${var.content_security_policy}"
        # Debug
        DEBUG                              = "${var.debug}"
        # SMTP
        SMTP_FQDN                          = "${var.smtp_fqdn}"
        SMTP_PORT                          = "${var.smtp_port}"
        # Additional (OIDC, LDAP, AAD, Custom Auth) – include if needed
        OIDC_ENABLE                        = "${var.oidc_enable}"
        OIDC_PROVIDER_URL                  = "${var.oidc_provider_url}"
        # ... add other auth vars as required (see compose)
      }
      resources {
        cpu    = 2000
        memory = 4096
      }
      service {
        name = "misp-core"
        port = "http"
        check {
          type     = "http"
          path     = "/users/heartbeat"
          interval = "30s"
          timeout  = "5s"
          check_restart {
            limit = 3
            grace = "60s"
          }
        }
      }
    }

    # ---------- MISP Guard (optional) ----------
    task "misp-guard" {
      driver = "docker"
      config {
        image = "ghcr.io/misp/misp-docker/misp-guard:${var.guard_running_tag}"
        ports = ["guard"]
        volumes = [
          "/opt/nomad/config/misp/guard/config.json:/config.json:ro",
          "/opt/nomad/volumes/misp/misp_guard_ca:/misp_guard_ca"
        ]
      }
      env {
        GUARD_PORT = "${var.guard_port}"
        GUARD_ARGS = "${var.guard_args}"
        TZ         = "${var.tz}"
      }
      resources {
        cpu    = 100
        memory = 128
      }
      service {
        name = "misp-guard"
        port = "guard"
        check {
          type     = "tcp"
          interval = "2m"
          timeout  = "5s"
          check_restart {
            limit = 3
            grace = "10s"
          }
        }
      }
    }
  }
}