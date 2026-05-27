# ==================== CLIENT CONFIGURATION ====================
data_dir = "/opt/consul/data"
server   = false

# Networking
bind_addr   = "{{ GetInterfaceIP \"eth0\" }}"
client_addr = "127.0.0.1"            # Only local access for security

# Addresses & ports (your snippet)
addresses {
  https = "0.0.0.0"                  # Allow HTTPS API from anywhere (use firewall)
}
ports {
  http  = 8500                       # Optional – keep for local CLI, or disable
  https = 8501
  grpc  = 8502
  serf_lan = 8301
}

# TLS (client side)
tls {
  defaults {
    ca_file = "/etc/consul/ssl/consul-agent-ca.pem"   # ← updated path
    verify_incoming        = false                     # clients don’t accept incoming RPC
    verify_outgoing        = true
    verify_server_hostname = true
  }
}

# Auto‑encrypt: client will request a certificate from servers
auto_encrypt {
  tls = true
}

# Gossip encryption – same key as servers
encrypt = "MRw/sVBSZUnK9sCfvoDPwQxEoMpo44p0mpdrX8qJqGE="

# ACLs – client needs agent token to register services
acl {
  enabled        = true
  default_policy = "deny"
  tokens {
    agent = "YOUR_AGENT_TOKEN"       # From `consul acl bootstrap` on a server
  }
}

# Join servers
retry_join = ["10.0.1.10", "10.0.1.11", "10.0.1.12"]

# Telemetry (optional)
telemetry {
  prometheus_retention_time = "24h"
}

# Logging
log_level = "INFO"
log_file  = "/var/log/consul/consul.log"