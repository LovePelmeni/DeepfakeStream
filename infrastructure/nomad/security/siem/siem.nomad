job "elastic-stack" {
  datacenters = ["dc1"]
  type        = "service"

  group "elastic" {
    # Shared network – all tasks inside the group can use localhost
    network {
      mode = "bridge"
      port "es_http" {
        static = 9200
        to     = 9200
      }
      port "kibana" {
        static = 5601
        to     = 5601
      }
      port "logstash_beats" {
        static = 5044
        to     = 5044
      }
      port "logstash_monitor" {
        static = 9600
        to     = 9600
      }
      port "fleet" {
        static = 8220
        to     = 8220
      }
    }

    # ---------- Elasticsearch ----------
    task "elasticsearch" {
      driver = "docker"

      config {
        image = "docker.elastic.co/elasticsearch/elasticsearch:8.15.0"
        ports = ["es_http"]
        ulimit {
          memlock = -1
          nofile  = 65536
        }
        volumes = [
          # Persistent data volume (create this directory on the host)
          "/opt/nomad/volumes/es_data:/usr/share/elasticsearch/data"
        ]
      }

      env {
        CLUSTER_NAME         = "production"
        DISCOVERY_TYPE       = "single-node"
        XPACK_SECURITY_ENABLED = "true"
        ELASTIC_PASSWORD     = "${ELASTIC_PASSWORD}"
        KIBANA_PASSWORD      = "${KIBANA_PASSWORD}"
        ES_JAVA_OPTS         = "-Xms4g -Xmx4g"
      }

      resources {
        cpu    = 2000
        memory = 4096
      }

      service {
        name = "elasticsearch"
        port = "es_http"
        check {
          type     = "http"
          path     = "/"
          interval = "30s"
          timeout  = "10s"
          # Basic auth check – note that password must be passed via env
          # For simplicity, we skip authentication in the check; or you can use a script.
          # Here we rely on the container's own healthcheck.
        }
      }
    }

    # ---------- Kibana ----------
    task "kibana" {
      driver = "docker"

      config {
        image = "docker.elastic.co/kibana/kibana:8.15.0"
        ports = ["kibana"]
      }

      env {
        ELASTICSEARCH_HOSTS                   = "http://localhost:9200"
        ELASTICSEARCH_USERNAME                = "kibana_system"
        ELASTICSEARCH_PASSWORD                = "${KIBANA_PASSWORD}"
        XPACK_FLEET_ENABLED                   = "true"
        XPACK_FLEET_AGENT_ID_VERIFICATION_ENABLED = "false"
        XPACK_FLEET_AGENTS_TLS_CHECK_DISABLED = "true"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "kibana"
        port = "kibana"
        check {
          type     = "http"
          path     = "/api/status"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }

    # ---------- Logstash ----------
    task "logstash" {
      driver = "docker"

      config {
        image = "docker.elastic.co/logstash/logstash:8.15.0"
        ports = ["logstash_beats", "logstash_monitor"]
        volumes = [
          # Mount pipeline and config from host – adjust paths as needed
          "/opt/nomad/config/logstash/pipeline:/usr/share/logstash/pipeline:ro",
          "/opt/nomad/config/logstash/config/logstash.yml:/usr/share/logstash/config/logstash.yml:ro"
        ]
      }

      env {
        XPACK_MONITORING_ENABLED = "false"
      }

      resources {
        cpu    = 500
        memory = 1024
      }

      service {
        name = "logstash"
        port = "logstash_monitor"
        check {
          type     = "http"
          path     = "/"
          interval = "30s"
          timeout  = "10s"
        }
      }
    }

    # ---------- Fleet Server ----------
    task "fleet-server" {
      driver = "docker"

      config {
        image = "docker.elastic.co/beats/elastic-agent:8.15.0"
        ports = ["fleet"]
      }

      env {
        FLEET_SERVER_ENABLE               = "true"
        FLEET_SERVER_ELASTICSEARCH_HOST   = "http://localhost:9200"
        FLEET_SERVER_ELASTICSEARCH_USERNAME = "elastic"
        FLEET_SERVER_ELASTICSEARCH_PASSWORD = "${ELASTIC_PASSWORD}"
        FLEET_SERVER_SERVICE_TOKEN        = "${FLEET_SERVER_SERVICE_TOKEN}"
        FLEET_SERVER_POLICY_ID            = "${FLEET_SERVER_POLICY_ID}"
      }

      resources {
        cpu    = 200
        memory = 512
      }
    }
  }
}