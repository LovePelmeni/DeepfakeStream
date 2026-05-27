from checkov.common.models.enums import CheckResult, CheckCategories
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck

class LimitInstanceCount(BaseResourceCheck):
    def __init__(self):
        name = "Yandex compute instance count must not exceed 5"
        id = "CUSTOM_INSTANCE_001"
        categories = [CheckCategories.CONVENTION]
        supported_resources = ["yandex_compute_instance"]
        super().__init__(name=name, id=id, categories=categories, supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        if "count" in conf:
            count = conf["count"]
            if isinstance(count, int) and count > 5:
                return CheckResult.FAILED
        return CheckResult.PASSED

check = LimitInstanceCount()