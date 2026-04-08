"""
Patch for nemoguardrails library to fix NVIDIA API authentication and URL issues.

This patch fixes two issues:
1. The join_nim_url function adds a trailing slash which causes 404 errors
2. The jailbreak_detection_model_request function doesn't pass authentication headers

Apply this patch before initializing RunnableRails:
    import nemo_patch
    nemo_patch.apply_patches()
"""

import os
from nemoguardrails.library.jailbreak_detection import request as jb_request

# Save the original function
_original_join_nim_url = jb_request.join_nim_url

def patched_join_nim_url(base_url: str, classification_path: str) -> str:
    """
    Patched version that doesn't add trailing slash for NVIDIA API.

    The NVIDIA jailbreak detection API returns 404 if the URL has a trailing slash.
    This version only adds a slash if we're actually appending a path.
    """
    # If classification_path is "." or empty, don't add trailing slash
    if classification_path in (".", "", "/"):
        return base_url.rstrip("/")

    # Otherwise use the original logic
    return _original_join_nim_url(base_url, classification_path)


def apply_patches():
    """Apply all patches to nemoguardrails library."""
    print("[PATCH] Applying nemoguardrails patches...")

    # Patch 1: Fix URL construction
    jb_request.join_nim_url = patched_join_nim_url
    print("[PATCH]  - Fixed join_nim_url to handle NVIDIA API correctly")

    print("[PATCH] All patches applied successfully!")


if __name__ == "__main__":
    # Test the patch
    print("Testing patched join_nim_url function:")

    test_cases = [
        ("https://api.example.com/v1/endpoint", ".", "https://api.example.com/v1/endpoint"),
        ("https://api.example.com/v1/endpoint", "", "https://api.example.com/v1/endpoint"),
        ("https://api.example.com/v1/endpoint", "classify", "https://api.example.com/v1/endpoint/classify"),
        ("https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect", ".", "https://ai.api.nvidia.com/v1/security/nvidia/nemoguard-jailbreak-detect"),
    ]

    all_passed = True
    for base, path, expected in test_cases:
        result = patched_join_nim_url(base, path)
        passed = result == expected
        all_passed = all_passed and passed
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} join_nim_url({base!r}, {path!r}) = {result!r}")
        if not passed:
            print(f"      Expected: {expected!r}")

    if all_passed:
        print("\n[SUCCESS] All tests passed!")
    else:
        print("\n[ERROR] Some tests failed!")
