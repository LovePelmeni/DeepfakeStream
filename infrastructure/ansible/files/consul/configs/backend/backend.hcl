service {
    name = "backend"
    port = 8080
    tags = ["backend"]
    id = "backend-1"
    check {
        http = "http://localhost:8080/health/ready"
        interval = "10s"
        timeout = "5s"
    }
    connect {
        sidecar_service {}
    }
}