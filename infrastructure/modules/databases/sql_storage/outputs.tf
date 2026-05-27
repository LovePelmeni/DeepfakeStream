output "database_config" {
  description = "Complete DB connection config"
  value = {
    PSQL_DB_HOST = module.pgsql.hosts[0].fqdn
    PSQL_DB_PORT = "6432"
    PSQL_DB_USER = module.pgsql.owners_data[0].name
    PSQL_DB_PASSWORD = module.pgsql.owners_data[0].password
    PSQL_DB_NAME = "dronespas_production"
    PSQL_DB_SSL_MODE = "require"
  }
  sensitive = true
}
