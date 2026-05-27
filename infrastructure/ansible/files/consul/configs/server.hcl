# ==================== SERVER CONFIGURATION ====================

# Basic paths
data_dir = "/opt/consul/data"

# Server mode (3‑node cluster)
server = true
bootstrap_expect = 3               # Must match number of servers

# Web UI (optional)
ui_config {
  enabled = true
}

# Networking
bind_addr   = "{{ GetInterfaceIP \"eth0\" }}"   # Use your private interface
client_addr = "0.0.0.0"          # Listen on all interfaces for API (restrict with firewall)

# Ports (explicit defaults, plus HTTPS on 8501)
ports {
  http   = 8500
  https  = 8501
  grpc   = 8502
  dns    = 8600
  server = 8300
  serf_lan = 8301
  serf_wan = 8302
}

# ==================== TLS / mTLS ====================
tls {
  defaults {
    ca_file   = "/etc/consul/ssl/consul-agent-ca.pem"
    cert_file = "/etc/consul/ssl/server-server1.pem"
    key_file  = "/etc/consul/ssl/server-server1-key.pem"

    verify_incoming        = true
    verify_outgoing        = true
    verify_server_hostname = true
  }
}

# ==================== Auto‑encrypt for clients ====================
auto_encrypt {
  allow_tls = true   # Clients can request TLS certificates
}

# ==================== Gossip encryption (same key on all agents) ====================
encrypt = "MRw/sVBSZUnK9sCfvoDPwQxEoMpo44p0mpdrX8qJqGE="

# ==================== ACLs ====================
acl {
  enabled        = true
  default_policy = "deny"
  enable_token_persistence = true
}

# ==================== Performance tuning ====================
performance {
  raft_multiplier = 1   # 1 for low‑latency LAN, 5 for high‑latency WAN
}

# ==================== Telemetry (Prometheus) ====================
telemetry {
  prometheus_retention_time = "24h"
  disable_hostname = true
}

# ==================== Logging ====================
log_level = "INFO"
log_file  = "/var/log/consul/consul.log"
log_rotate_bytes = 104857600      # 100 MB
log_rotate_duration = "24h"

# ==================== Joining other servers ====================
retry_join = ["10.0.1.10", "10.0.1.11", "10.0.1.12"]   # IPs of all server nodes