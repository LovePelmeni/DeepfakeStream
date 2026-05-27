output "waf_profile_id" {
  description = "WAF profile ID"
  value       = try(yandex_sws_waf_profile.main[0].id, null)
}

output "waf_profile_name" {
  description = "WAF profile name"
  value       = try(yandex_sws_waf_profile.main[0].name, null)
}

output "waf_rules_count" {
  description = "Number of enabled WAF rules (not available - returns 0)"
  value       = 0
}

output "ip_whitelist_rule_id" {
  description = "IP whitelist rule ID (not implemented - use SmartWeb Security)"
  value       = null
}

output "ip_blacklist_rule_id" {
  description = "IP blacklist rule ID (not implemented - use SmartWeb Security)"
  value       = null
}

output "waf_profile_config" {
  description = "WAF profile configuration summary"
  value = try({
    id                 = yandex_sws_waf_profile.main[0].id
    name               = yandex_sws_waf_profile.main[0].name
    description        = yandex_sws_waf_profile.main[0].description
    paranoia_level     = var.enable_owasp_paranoia_level
    anomaly_threshold  = var.anomaly_threshold
    mode               = var.waf_mode
    enabled            = var.enabled
  }, null)
}

output "exclusion_rules_count" {
  description = "Number of exclusion rules"
  value       = length(var.exclusion_rules)
}

output "is_enabled" {
  description = "Whether WAF is enabled"
  value       = var.enabled
}

output "core_rule_set_version" {
  description = "OWASP Core Rule Set version"
  value       = "4.0.0"
}

output "request_body_analysis" {
  description = "Request body analysis configuration"
  value = {
    enabled             = true
    size_limit_kb       = 8
    size_limit_action   = "DENY"
  }
}