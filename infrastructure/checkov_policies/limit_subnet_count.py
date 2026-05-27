from checkov.common.models.enums import CheckResult, CheckCategories
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

class LimitSubnetCount(BaseResourceCheck):
    def __init__(self):
        name = "Yandex subnet count must not exceed 3"
        id = "CUSTOM_SUBNET_001"
        categories = [CheckCategories.CONVENTION]
        supported_resources = ["yandex_vpc_subnet"]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if "count" in conf:
            count = conf["count"]
            if isinstance(count, int) and count > 3:
                return CheckResult.FAILED
        return CheckResult.PASSED

check = LimitSubnetCount()