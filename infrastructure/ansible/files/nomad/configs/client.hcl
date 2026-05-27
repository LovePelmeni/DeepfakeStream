client {
  host_volume "vegamaps_redis_data"     { path = "/opt/nomad/volumes/vegamaps_redis_data" }
  host_volume "vegamaps_ssl"            { path = "/opt/nomad/volumes/vegamaps_ssl" }
  host_volume "vegamaps_recordings"     { path = "/opt/nomad/volumes/vegamaps_recordings" }
  host_volume "vegamaps_hls"            { path = "/opt/nomad/volumes/vegamaps_hls" }
  host_volume "prometheus_data"         { path = "/opt/nomad/volumes/prometheus_data" }
  host_volume "prometheus_config"       { path = "/opt/nomad/volumes/prometheus_config" }
  host_volume "prometheus_rules"        { path = "/opt/nomad/volumes/prometheus_rules" }
  host_volume "loki_data"               { path = "/opt/nomad/volumes/loki_data" }
  host_volume "loki_config"             { path = "/opt/nomad/volumes/loki_config" }
  host_volume "alloy_data"              { path = "/opt/nomad/volumes/alloy_data" }
  host_volume "alloy_config"            { path = "/opt/nomad/volumes/alloy_config" }
  host_volume "tempo_data"              { path = "/opt/nomad/volumes/tempo_data" }
  host_volume "tempo_config"            { path = "/opt/nomad/volumes/tempo_config" }
  host_volume "alertmanager_data"       { path = "/opt/nomad/volumes/alertmanager_data" }
  host_volume "alertmanager_config"     { path = "/opt/nomad/volumes/alertmanager_config" }
  host_volume "pyroscope_data"          { path = "/opt/nomad/volumes/pyroscope_data" }
  host_volume "mysql_bugsink_data"      { path = "/opt/nomad/volumes/mysql_bugsink_data" }
  host_volume "bugsink_data"            { path = "/opt/nomad/volumes/bugsink_data" }
  host_volume "grafana_data"            { path = "/opt/nomad/volumes/grafana_data" }
  host_volume "grafana_provisioning"    { path = "/opt/nomad/volumes/grafana_provisioning" }
}