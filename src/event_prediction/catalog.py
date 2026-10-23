"""Stable observable event vocabulary."""

from types import MappingProxyType

EVENT_DESCRIPTIONS = MappingProxyType(
    {
        "api_interface_modified": "An API interface was modified.",
        "dependency_upgraded": "A software dependency was upgraded.",
        "compatibility_checked": "Compatibility with dependent components was checked.",
        "ci_failed": "Continuous integration reported a failure.",
        "bug_fixed": "A defect discovered during delivery was fixed.",
        "integration_test_passed": "Integration tests completed successfully.",
        "review_approved": "A reviewer approved the change.",
        "deployed": "The change was deployed.",
        "rolled_back": "A deployment was reverted.",
    }
)
EVENT_TYPES = tuple(EVENT_DESCRIPTIONS)


def validate_event_code(code: str) -> str:
    if code not in EVENT_DESCRIPTIONS:
        allowed = ", ".join(EVENT_TYPES)
        raise ValueError(f"unknown event type {code!r}; expected one of: {allowed}")
    return code

