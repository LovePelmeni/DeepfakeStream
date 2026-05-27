from checkov.common.models.enums import CheckResult, CheckCategories
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

class LimitVpcCount(BaseResourceCheck):
    def __init__(self):
        name = "Yandex VPC count must not exceed 2"
        id = "CUSTOM_VPC_001"
        categories = [CheckCategories.CONVENTION]
        supported_resources = ["yandex_vpc_network"]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if "count" in conf:
            count = conf["count"]
            if isinstance(count, int) and count > 2:
                return CheckResult.FAILED
        return CheckResult.PASSED

check = LimitVpcCount()