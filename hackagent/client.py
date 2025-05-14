import ssl
from typing import Any, Optional, Union

import httpx
from attrs import define, evolve, field


# New: Custom HTTP client to fix multipart boundary issues
class MultipartFixClient(httpx.Client):
    def request(
        self, method: str, url: Union[str, httpx.URL], **kwargs: Any
    ) -> httpx.Response:
        # Check if this is a multipart request being prepared by openapi-python-client
        # The key indicators are the presence of 'files' in kwargs and a manually set
        # Content-Type header that might be missing the boundary.
        headers = kwargs.get("headers")
        if kwargs.get("files") is not None and headers is not None:
            content_type = headers.get("Content-Type")
            # If Content-Type is exactly "multipart/form-data" (without a boundary),
            # httpx might not overwrite it correctly. We remove it to let httpx
            # generate the full header including the boundary from the 'files' kwarg.
            if content_type == "multipart/form-data":
                # Create a new dict for headers to avoid modifying the original in an unexpected way
                new_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
                kwargs["headers"] = new_headers
            # If Content-Type includes a boundary but is still problematic,
            # more specific checks might be needed, but usually httpx handles it if 'files' is present
            # and no conflicting Content-Type is explicitly set without a boundary.

        return super().request(method, url, **kwargs)


# New: Async version of the custom client
class AsyncMultipartFixClient(httpx.AsyncClient):
    async def request(
        self, method: str, url: Union[str, httpx.URL], **kwargs: Any
    ) -> httpx.Response:
        headers = kwargs.get("headers")
        if kwargs.get("files") is not None and headers is not None:
            content_type = headers.get("Content-Type")
            if content_type == "multipart/form-data":
                new_headers = {k: v for k, v in headers.items() if k != "Content-Type"}
                kwargs["headers"] = new_headers
        return await super().request(method, url, **kwargs)


@define
class Client:
    """A class for keeping track of data related to the API

    The following are accepted as keyword arguments and will be used to construct httpx Clients internally:

        ``base_url``: The base URL for the API, all requests are made to a relative path to this URL

        ``cookies``: A dictionary of cookies to be sent with every request

        ``headers``: A dictionary of headers to be sent with every request

        ``timeout``: The maximum amount of a time a request can take. API functions will raise
        httpx.TimeoutException if this is exceeded.

        ``verify_ssl``: Whether or not to verify the SSL certificate of the API server. This should be True in production,
        but can be set to False for testing purposes.

        ``follow_redirects``: Whether or not to follow redirects. Default value is False.

        ``httpx_args``: A dictionary of additional arguments to be passed to the ``httpx.Client`` and ``httpx.AsyncClient`` constructor.


    Attributes:
        raise_on_unexpected_status: Whether or not to raise an errors.UnexpectedStatus if the API returns a
            status code that was not documented in the source OpenAPI document. Can also be provided as a keyword
            argument to the constructor.
    """

    raise_on_unexpected_status: bool = field(default=False, kw_only=True)
    _base_url: str = field(alias="base_url")
    _cookies: dict[str, str] = field(factory=dict, kw_only=True, alias="cookies")
    _headers: dict[str, str] = field(factory=dict, kw_only=True, alias="headers")
    _timeout: Optional[httpx.Timeout] = field(
        default=None, kw_only=True, alias="timeout"
    )
    _verify_ssl: Union[str, bool, ssl.SSLContext] = field(
        default=True, kw_only=True, alias="verify_ssl"
    )
    _follow_redirects: bool = field(
        default=False, kw_only=True, alias="follow_redirects"
    )
    _httpx_args: dict[str, Any] = field(factory=dict, kw_only=True, alias="httpx_args")
    _client: Optional[httpx.Client] = field(default=None, init=False)
    _async_client: Optional[httpx.AsyncClient] = field(default=None, init=False)

    def with_headers(self, headers: dict[str, str]) -> "Client":
        """Get a new client matching this one with additional headers"""
        if self._client is not None:
            self._client.headers.update(headers)
        if self._async_client is not None:
            self._async_client.headers.update(headers)
        return evolve(self, headers={**self._headers, **headers})

    def with_cookies(self, cookies: dict[str, str]) -> "Client":
        """Get a new client matching this one with additional cookies"""
        if self._client is not None:
            self._client.cookies.update(cookies)
        if self._async_client is not None:
            self._async_client.cookies.update(cookies)
        return evolve(self, cookies={**self._cookies, **cookies})

    def with_timeout(self, timeout: httpx.Timeout) -> "Client":
        """Get a new client matching this one with a new timeout (in seconds)"""
        if self._client is not None:
            self._client.timeout = timeout
        if self._async_client is not None:
            self._async_client.timeout = timeout
        return evolve(self, timeout=timeout)

    def set_httpx_client(self, client: httpx.Client) -> "Client":
        """Manually set the underlying httpx.Client

        **NOTE**: This will override any other settings on the client, including cookies, headers, and timeout.
        """
        self._client = client
        return self

    def get_httpx_client(self) -> httpx.Client:
        """Get the underlying httpx.Client, constructing a new one if not previously set"""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._client

    def __enter__(self) -> "Client":
        """Enter a context manager for self.client—you cannot enter twice (see httpx docs)"""
        self.get_httpx_client().__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        """Exit a context manager for internal httpx.Client (see httpx docs)"""
        self.get_httpx_client().__exit__(*args, **kwargs)

    def set_async_httpx_client(self, async_client: httpx.AsyncClient) -> "Client":
        """Manually the underlying httpx.AsyncClient

        **NOTE**: This will override any other settings on the client, including cookies, headers, and timeout.
        """
        self._async_client = async_client
        return self

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        """Get the underlying httpx.AsyncClient, constructing a new one if not previously set"""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                cookies=self._cookies,
                headers=self._headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._async_client

    async def __aenter__(self) -> "Client":
        """Enter a context manager for underlying httpx.AsyncClient—you cannot enter twice (see httpx docs)"""
        await self.get_async_httpx_client().__aenter__()
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        """Exit a context manager for underlying httpx.AsyncClient (see httpx docs)"""
        await self.get_async_httpx_client().__aexit__(*args, **kwargs)


@define
class AuthenticatedClient:
    """A Client which has been authenticated for use on secured endpoints

    The following are accepted as keyword arguments and will be used to construct httpx Clients internally:

        ``base_url``: The base URL for the API, all requests are made to a relative path to this URL

        ``cookies``: A dictionary of cookies to be sent with every request

        ``headers``: A dictionary of headers to be sent with every request

        ``timeout``: The maximum amount of a time a request can take. API functions will raise
        httpx.TimeoutException if this is exceeded.

        ``verify_ssl``: Whether or not to verify the SSL certificate of the API server. This should be True in production,
        but can be set to False for testing purposes.

        ``follow_redirects``: Whether or not to follow redirects. Default value is False.

        ``httpx_args``: A dictionary of additional arguments to be passed to the ``httpx.Client`` and ``httpx.AsyncClient`` constructor.


    Attributes:
        raise_on_unexpected_status: Whether or not to raise an errors.UnexpectedStatus if the API returns a
            status code that was not documented in the source OpenAPI document. Can also be provided as a keyword
            argument to the constructor.
        token: The token to use for authentication
        prefix: The prefix to use for the Authorization header
        auth_header_name: The name of the Authorization header
    """

    token: str
    raise_on_unexpected_status: bool = field(default=False, kw_only=True)
    _base_url: str = field(
        default="https://hackagent-webapp-260146888364.europe-west1.run.app/",
        alias="base_url",
    )
    _cookies: dict[str, str] = field(factory=dict, kw_only=True, alias="cookies")
    _headers: dict[str, str] = field(factory=dict, kw_only=True, alias="headers")
    _timeout: Optional[httpx.Timeout] = field(
        default=None, kw_only=True, alias="timeout"
    )
    _verify_ssl: Union[str, bool, ssl.SSLContext] = field(
        default=True, kw_only=True, alias="verify_ssl"
    )
    _follow_redirects: bool = field(
        default=False, kw_only=True, alias="follow_redirects"
    )
    _httpx_args: dict[str, Any] = field(factory=dict, kw_only=True, alias="httpx_args")
    _client: Optional[httpx.Client] = field(default=None, init=False)
    _async_client: Optional[httpx.AsyncClient] = field(default=None, init=False)

    prefix: str = "Bearer"
    auth_header_name: str = "Authorization"

    def with_headers(self, headers: dict[str, str]) -> "AuthenticatedClient":
        """Get a new client matching this one with additional headers"""
        if self._client is not None:
            self._client.headers.update(headers)
        if self._async_client is not None:
            self._async_client.headers.update(headers)
        return evolve(self, headers={**self._headers, **headers})

    def with_cookies(self, cookies: dict[str, str]) -> "AuthenticatedClient":
        """Get a new client matching this one with additional cookies"""
        if self._client is not None:
            self._client.cookies.update(cookies)
        if self._async_client is not None:
            self._async_client.cookies.update(cookies)
        return evolve(self, cookies={**self._cookies, **cookies})

    def with_timeout(self, timeout: httpx.Timeout) -> "AuthenticatedClient":
        """Get a new client matching this one with a new timeout (in seconds)"""
        if self._client is not None:
            self._client.timeout = timeout
        if self._async_client is not None:
            self._async_client.timeout = timeout
        return evolve(self, timeout=timeout)

    def set_httpx_client(self, client: httpx.Client) -> "AuthenticatedClient":
        """Manually set the underlying httpx.Client. Should be MultipartFixClient or compatible."""
        if not isinstance(client, MultipartFixClient):
            # Or log a warning, or be more flexible depending on desired strictness
            pass  # Consider raising TypeError if strict type is required
        self._client = client
        return self

    def get_httpx_client(self) -> httpx.Client:
        """Get the underlying httpx.Client, constructing a new MultipartFixClient if not previously set"""
        if self._client is None:
            # Prepare auth headers to be part of the initial headers for MultipartFixClient
            request_headers = (
                self._headers.copy()
            )  # Start with base headers passed to AuthenticatedClient
            auth_value = f"{self.prefix} {self.token}" if self.prefix else self.token
            request_headers[self.auth_header_name] = auth_value

            self._client = MultipartFixClient(  # Use the custom client
                base_url=self._base_url,
                cookies=self._cookies,
                headers=request_headers,  # Pass combined headers
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._client

    def __enter__(self) -> "AuthenticatedClient":
        self.get_httpx_client().__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self.get_httpx_client().__exit__(*args, **kwargs)

    def set_async_httpx_client(
        self, async_client: httpx.AsyncClient
    ) -> "AuthenticatedClient":
        """Manually set the underlying httpx.AsyncClient. Should be AsyncMultipartFixClient or compatible."""
        if not isinstance(async_client, AsyncMultipartFixClient):
            pass  # Consider raising TypeError
        self._async_client = async_client
        return self

    def get_async_httpx_client(self) -> httpx.AsyncClient:
        """Get the underlying httpx.AsyncClient, constructing new AsyncMultipartFixClient if not previously set"""
        if self._async_client is None:
            request_headers = self._headers.copy()
            auth_value = f"{self.prefix} {self.token}" if self.prefix else self.token
            request_headers[self.auth_header_name] = auth_value

            self._async_client = AsyncMultipartFixClient(  # Use the custom async client
                base_url=self._base_url,
                cookies=self._cookies,
                headers=request_headers,
                timeout=self._timeout,
                verify=self._verify_ssl,
                follow_redirects=self._follow_redirects,
                **self._httpx_args,
            )
        return self._async_client

    async def __aenter__(self) -> "AuthenticatedClient":
        await self.get_async_httpx_client().__aenter__()
        return self

    async def __aexit__(self, *args: Any, **kwargs: Any) -> None:
        await self.get_async_httpx_client().__aexit__(*args, **kwargs)
