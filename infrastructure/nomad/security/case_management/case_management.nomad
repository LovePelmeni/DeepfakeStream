job "soar" {
  datacenters = ["dc1"]
  type        = "service"

  # Variables – define defaults or pass via -var / var-file
  variables {
    elastic_password   = "changeme"
    thehive_jwt_secret = "changeme"
    thehive_admin_pwd  = "changeme"
    cortex_secret_key  = "changeme"
    cortex_admin_pwd   = "changeme"
  }

  group "soar" {
    # Shared network – all tasks can use localhost
    network {
      mode = "bridge"
      port "es_http" {
        static = 9201
        to     = 9200
      }
      port "es_transport" {
        static = 9301
        to     = 9300
      }
      port "cassandra" {
        static = 9042
        to     = 9042
      }
      port "thehive" {
        static = 9000
        to     = 9000
      }
      port "cortex" {
        static = 9001
        to     = 9001
      }
    }

    # ---------- Elasticsearch 7 ----------
    task "elasticsearch7" {
      driver = "docker"

      config {
        image = "docker.elastic.co/elasticsearch/elasticsearch:7.17.23"
        ports = ["es_http", "es_transport"]
        ulimit {
          memlock = -1
          nofile  = 65536
        }
        volumes = [
          "/opt/nomad/volumes/soar/es7_data:/usr/share/elasticsearch/data"
        ]
      }

      env {
        CLUSTER_NAME         = "soar"
        DISCOVERY_TYPE       = "single-node"
        XPACK_SECURITY_ENABLED = "true"
        ELASTIC_PASSWORD     = "${var.elastic_password}"
        ES_JAVA_OPTS         = "-Xms1g -Xmx1g"
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      service {
        name = "elasticsearch7"
        port = "es_http"
        check {
          type     = "http"
          path     = "/"
          interval = "30s"
          timeout  = "10s"
          # Authentication not needed for check; but compose uses curl with auth.
          # We rely on container healthcheck or a custom script.
        }
      }
    }

    # ---------- Cassandra ----------
    task "cassandra" {
      driver = "docker"

      config {
        image = "cassandra:4.1"
        ports = ["cassandra"]
        volumes = [
          "/opt/nomad/volumes/soar/cassandra_data:/var/lib/cassandra"
        ]
      }

      env {
        CASSANDRA_CLUSTER_NAME = "SOARCluster"
        MAX_HEAP_SIZE          = "512M"
        HEAP_NEWSIZE           = "100M"
      }

      resources {
        cpu    = 1000
        memory = 1024
      }

      service {
        name = "cassandra"
        port = "cassandra"
        check {
          type     = "tcp"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }

    # ---------- Cortex ----------
    task "cortex" {
      driver = "docker"

      config {
        image = "thehiveproject/cortex:5.5.3"
        ports = ["cortex"]
      }

      env {
        ELASTICSEARCH_ADDRESSES              = "https://localhost:9200"
        ELASTICSEARCH_USERNAME               = "elastic"
        ELASTICSEARCH_PASSWORD               = "${var.elastic_password}"
        PLAY_HTTP_SECRET_KEY                 = "${var.cortex_secret_key}"
        AUTH_LOCAL_USER_DEFAULT_PASSWORD     = "${var.cortex_admin_pwd}"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "cortex"
        port = "cortex"
        check {
          type     = "http"
          path     = "/api/status"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }

    # ---------- TheHive ----------
    task "thehive" {
      driver = "docker"

      config {
        image = "thehiveproject/thehive:5.5.3"
        ports = ["thehive"]
      }

      env {
        ELASTICSEARCH_ADDRESSES = "https://localhost:9200"
        ELASTICSEARCH_USERNAME  = "elastic"
        ELASTICSEARCH_PASSWORD  = "${var.elastic_password}"
        CASSANDRA_ADDRESSES     = "localhost:9042"
        JWT_SECRET              = "${var.thehive_jwt_secret}"
        THEHIVE_ADMIN_PASSWORD  = "${var.thehive_admin_pwd}"
      }

      resources {
        cpu    = 1000
        memory = 2048
      }

      service {
        name = "thehive"
        port = "thehive"
        check {
          type     = "http"
          path     = "/api/status"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }
  }
}