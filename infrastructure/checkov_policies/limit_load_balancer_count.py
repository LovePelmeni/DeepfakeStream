from checkov.common.models.enums import CheckResult, CheckCategories
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

class LimitLbCount(BaseResourceCheck):
    def __init__(self):
        name = "Yandex load balancer count must not exceed 1"
        id = "CUSTOM_LB_001"
        categories = [CheckCategories.CONVENTION]
        supported_resources = ["yandex_lb_network_load_balancer", "yandex_alb_load_balancer"]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if "count" in conf:
            count = conf["count"]
            if isinstance(count, int) and count > 1:
                return CheckResult.FAILED
        return CheckResult.PASSED

check = LimitLbCount()