import unittest
from unittest.mock import MagicMock, AsyncMock
from http import HTTPStatus
import httpx
import asyncio

from hackagent.api.judge.judge_create import sync_detailed, asyncio_detailed
from hackagent.client import AuthenticatedClient
from hackagent.types import Response
from hackagent import errors


class TestJudgeAPI(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock(spec=AuthenticatedClient)
        self.mock_client.raise_on_unexpected_status = True
        self.mock_httpx_client = MagicMock()
        self.mock_async_httpx_client = MagicMock()
        self.mock_client.get_httpx_client.return_value = self.mock_httpx_client
        self.mock_client.get_async_httpx_client.return_value = (
            self.mock_async_httpx_client
        )

    def test_sync_detailed_success(self):
        mock_response = httpx.Response(
            HTTPStatus.OK,
            content=b"Success",
            headers={"Content-Type": "application/json"},
        )
        self.mock_httpx_client.request.return_value = mock_response

        response = sync_detailed(client=self.mock_client)

        self.mock_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.content, b"Success")
        self.assertIsNone(response.parsed)  # As _parse_response returns None for 200

    def test_sync_detailed_unexpected_status(self):
        mock_response = httpx.Response(
            HTTPStatus.BAD_REQUEST,
            content=b"Error",
            headers={"Content-Type": "application/json"},
        )
        self.mock_httpx_client.request.return_value = mock_response

        with self.assertRaises(errors.UnexpectedStatus) as cm:
            sync_detailed(client=self.mock_client)

        self.assertEqual(cm.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(cm.exception.content, b"Error")
        self.mock_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )

    def test_sync_detailed_unexpected_status_no_raise(self):
        self.mock_client.raise_on_unexpected_status = False
        mock_response = httpx.Response(
            HTTPStatus.BAD_REQUEST,
            content=b"Error",
            headers={"Content-Type": "application/json"},
        )
        self.mock_httpx_client.request.return_value = mock_response

        response = sync_detailed(client=self.mock_client)

        self.mock_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )
        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIsNone(response.parsed)

    def test_asyncio_detailed_success(self):
        mock_async_response = MagicMock(spec=httpx.Response)
        mock_async_response.status_code = HTTPStatus.OK
        mock_async_response.content = b"Async Success"
        mock_async_response.headers = {"Content-Type": "application/json"}

        self.mock_async_httpx_client.request = AsyncMock(
            return_value=mock_async_response
        )

        async def run_test():
            return await asyncio_detailed(client=self.mock_client)

        response = asyncio.run(run_test())

        self.mock_async_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )
        self.assertIsInstance(response, Response)
        self.assertEqual(response.status_code, HTTPStatus.OK)
        self.assertEqual(response.content, b"Async Success")
        self.assertIsNone(response.parsed)

    def test_asyncio_detailed_unexpected_status(self):
        mock_async_response = MagicMock(spec=httpx.Response)
        mock_async_response.status_code = HTTPStatus.BAD_REQUEST
        mock_async_response.content = b"Async Error"
        mock_async_response.headers = {"Content-Type": "application/json"}

        self.mock_async_httpx_client.request = AsyncMock(
            return_value=mock_async_response
        )

        async def run_test():
            with self.assertRaises(errors.UnexpectedStatus) as cm:
                await asyncio_detailed(client=self.mock_client)
            return cm

        cm = asyncio.run(run_test())

        self.assertEqual(cm.exception.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(cm.exception.content, b"Async Error")
        self.mock_async_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )

    def test_asyncio_detailed_unexpected_status_no_raise(self):
        self.mock_client.raise_on_unexpected_status = False
        mock_async_response = MagicMock(spec=httpx.Response)
        mock_async_response.status_code = HTTPStatus.BAD_REQUEST
        mock_async_response.content = b"Async Error"
        mock_async_response.headers = {"Content-Type": "application/json"}

        self.mock_async_httpx_client.request = AsyncMock(
            return_value=mock_async_response
        )

        async def run_test():
            return await asyncio_detailed(client=self.mock_client)

        response = asyncio.run(run_test())

        self.mock_async_httpx_client.request.assert_called_once_with(
            method="post", url="/api/judge"
        )
        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIsNone(response.parsed)


if __name__ == "__main__":
    unittest.main()
