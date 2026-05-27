# WAF Profile using Yandex SWS (Smart Web Security)
resource "yandex_sws_waf_profile" "main" {
  count = var.enabled ? 1 : 0
  
  name        = "${var.name_prefix}-profile"
  folder_id   = var.folder_id
  description = var.description
  
  labels = merge(var.labels, {
    service = "waf"
  })

  # Core Rule Set (OWASP)
  core_rule_set {
    inbound_anomaly_score = var.anomaly_threshold
    paranoia_level        = var.enable_owasp_paranoia_level
    
    rule_set {
      name    = "OWASP Core Ruleset"
      version = "4.0.0"
    }
  }

  # Request body analysis
  analyze_request_body {
    is_enabled        = true
    size_limit        = 8  # KB
    size_limit_action = "DENY"
  }
}