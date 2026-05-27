job "traefik-crowdsec" {
  datacenters = ["dc1"]
  type        = "service"

  group "proxy" {
    # Network configuration – expose ports on the host
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
    }

    # ---------- Traefik ----------
    task "traefik" {
      driver = "docker"

      config {
        image   = "traefik:v3.3"
        command = ["--configFile=/etc/traefik/traefik.yml"]
        ports   = ["http", "https"]
        volumes = [
          # Mount static config file (must exist on host)
          "/opt/nomad/config/traefik/traefik.yml:/etc/traefik/traefik.yml:ro",
          # Mount certs directory
          "/opt/nomad/config/traefik/certs:/etc/traefik/certs:ro",
          # Persistent log volume (host directory)
          "/opt/nomad/volumes/traefik-logs:/var/log/traefik"
        ]
      }

      env {
        # Optional: provide Nomad token if ACLs are enabled
        NOMAD_TOKEN = ""
      }

      resources {
        cpu    = 200
        memory = 256
      }

      service {
        name = "traefik"
        port = "http"
        tags = ["traefik.enable=true"]   # Optional: self‑registration for the dashboard
      }
    }

    # ---------- Crowdsec ----------
    task "crowdsec" {
      driver = "docker"

      config {
        image = "crowdsecurity/crowdsec:v1.6.6"
        volumes = [
          "/opt/nomad/volumes/crowdsec-db:/var/lib/crowdsec/data",
          "/opt/nomad/volumes/crowdsec-config:/etc/crowdsec",
          # Read Traefik logs from the host directory (same as traefik-logs)
          "/opt/nomad/volumes/traefik-logs:/var/log/traefik:ro"
        ]
      }

      env {
        COLLECTIONS        = "crowdsecurity/linux crowdsecurity/traefik crowdsecurity/http-cve crowdsecurity/appsec-generic-rules"
        CUSTOM_HOSTNAME    = "crowdsec"
        BOUNCER_KEY_PLUGIN = "${BOUNCER_KEY_PLUGIN}"   # from var file or environment
      }

      resources {
        cpu    = 200
        memory = 512
      }
    }

    # ---------- Bouncer (Traefik plugin) ----------
    task "bouncer" {
      driver = "docker"

      config {
        image = "docker.io/fbonalair/traefik-crowdsec-bouncer:v3.0.0"
      }

      env {
        CROWDSEC_BOUNCER_API_KEY = "${BOUNCER_KEY_PLUGIN}"
        CROWDSEC_AGENT_HOST      = "crowdsec:8080"
        GIN_MODE                 = "release"
      }

      resources {
        cpu    = 50
        memory = 128
      }

      service {
        name = "crowdsec-bouncer"
        port = "8081"
        tags = [
          "traefik.enable=true",
          "traefik.http.routers.bouncer.rule=Host(`bouncer.example.com`)",
          "traefik.http.routers.bouncer.entrypoints=websecure",
          "traefik.http.routers.bouncer.tls.certresolver=letsencrypt",
          "traefik.http.services.bouncer.loadbalancer.server.port=8081"
        ]
      }
    }
  }
}