"""Transport boundary for non-idempotent GLEE API operations."""

from __future__ import annotations

from typing import Any

from glee_sdk import GleeClient
from glee_sdk.client import _raise_for_response


NON_REPLAYING_POST_TRANSPORT_CONTRACT = "glee-non-replaying-post-transport-v1"


class NonReplayingGleeClient(GleeClient):
    """Retain SDK behavior for safe methods while making each POST exactly once."""

    def _request(self, method: str, path: str, **kwargs: Any) -> dict | list:
        if method.upper() != "POST":
            return super()._request(method, path, **kwargs)
        url = f"{self.api_url}{path}"
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.request(method, url, **kwargs)
        _raise_for_response(response)
        if response.status_code == 204:
            return {}
        return response.json()
