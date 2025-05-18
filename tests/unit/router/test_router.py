import unittest
from unittest.mock import patch, MagicMock
import uuid

# Assuming AgentTypeEnum and other necessary enums/models are accessible
# We might need to adjust imports based on the actual structure of hackagent.models
from hackagent.models import AgentTypeEnum, Agent as BackendAgentModel, UserAPIKey
from hackagent.router.router import AgentRouter
from hackagent.client import AuthenticatedClient


class TestAgentRouterInitialization(unittest.TestCase):
    @patch("hackagent.router.router.key_list")
    @patch("hackagent.router.router.agent_list")
    @patch("hackagent.router.router.agent_create")
    @patch("hackagent.router.router.agent_partial_update")
    @patch("hackagent.router.router.LiteLLMAgentAdapter", autospec=True)
    @patch("hackagent.router.router.ADKAgentAdapter", autospec=True)
    @patch("hackagent.router.router.AGENT_TYPE_TO_ADAPTER_MAP", new_callable=dict)
    def test_agent_router_init_creates_new_agent_if_not_exists(
        self,
        MockAgentMap,
        MockADKAdapter,
        MockLiteLLMAdapter,
        mock_agent_partial_update,
        mock_agent_create,
        mock_agent_list,
        mock_key_list,
    ):
        # --- MOCK SETUP ---
        MockAgentMap[AgentTypeEnum.GOOGLE_ADK] = MockADKAdapter
        MockAgentMap[AgentTypeEnum.LITELMM] = MockLiteLLMAdapter

        # Set the __name__ attribute for the mocked classes for logging purposes
        MockADKAdapter.__name__ = "ADKAgentAdapter"
        MockLiteLLMAdapter.__name__ = "LiteLLMAgentAdapter"

        # Optional: Add a debug print/log for the mock in the test
        # print(f"DEBUG_TEST: MockADKAdapter in test is: {MockADKAdapter}, id: {id(MockADKAdapter)}")

        # Mock AuthenticatedClient
        mock_client = MagicMock(spec=AuthenticatedClient)
        mock_client.token = "test_token_prefix_12345"

        # Mock key_list response
        mock_org_id = uuid.uuid4()
        mock_user_id = 123
        mock_api_key_obj = MagicMock(spec=UserAPIKey)
        mock_api_key_obj.prefix = "test_token_prefix_"
        mock_api_key_obj.organization = mock_org_id
        mock_api_key_obj.user = mock_user_id

        mock_key_list_response = MagicMock()
        mock_key_list_response.status_code = 200
        mock_key_list_response.parsed = MagicMock()
        mock_key_list_response.parsed.results = [mock_api_key_obj]
        mock_key_list.sync_detailed.return_value = mock_key_list_response

        # Mock agent_list response (agent does not exist)
        mock_agent_list_response = MagicMock()
        mock_agent_list_response.status_code = 200
        mock_agent_list_response.parsed = MagicMock()
        mock_agent_list_response.parsed.results = []
        mock_agent_list_response.parsed.next_ = None
        mock_agent_list.sync_detailed.return_value = mock_agent_list_response

        # Mock agent_create response
        mock_created_agent_id = uuid.uuid4()
        mock_backend_agent_from_create = MagicMock(spec=BackendAgentModel)
        mock_backend_agent_from_create.id = mock_created_agent_id
        mock_backend_agent_from_create.name = "TestAgent"
        mock_backend_agent_from_create.agent_type = AgentTypeEnum.GOOGLE_ADK
        mock_backend_agent_from_create.endpoint = "http://fake-agent-endpoint.com"
        mock_backend_agent_from_create.metadata = {"initial_meta": "value"}
        mock_backend_agent_from_create.organization = mock_org_id

        mock_agent_create_response = MagicMock()
        mock_agent_create_response.status_code = 201
        mock_agent_create_response.parsed = mock_backend_agent_from_create
        mock_agent_create.sync_detailed.return_value = mock_agent_create_response

        # --- TEST PARAMETERS ---
        agent_name = "TestAgent"
        agent_type = AgentTypeEnum.GOOGLE_ADK
        agent_endpoint = "http://fake-agent-endpoint.com"
        agent_metadata = {"initial_meta": "value"}
        adapter_op_config = {"user_id": "test_user_from_op_config"}

        # --- EXECUTE ---
        router = AgentRouter(
            client=mock_client,
            name=agent_name,
            agent_type=agent_type,
            endpoint=agent_endpoint,
            metadata=agent_metadata,
            adapter_operational_config=adapter_op_config,
            overwrite_metadata=True,
        )

        # --- ASSERTIONS ---
        self.assertEqual(mock_key_list.sync_detailed.call_count, 2)
        mock_agent_list.sync_detailed.assert_called_once()
        mock_agent_create.sync_detailed.assert_called_once()
        create_call_args_kwargs = mock_agent_create.sync_detailed.call_args[1]
        self.assertEqual(create_call_args_kwargs["client"], mock_client)
        agent_request_body = create_call_args_kwargs["body"]
        self.assertEqual(agent_request_body.name, agent_name)
        self.assertEqual(agent_request_body.agent_type, agent_type)
        self.assertEqual(agent_request_body.endpoint, agent_endpoint)
        self.assertEqual(agent_request_body.metadata, agent_metadata)
        self.assertEqual(agent_request_body.organization, mock_org_id)

        mock_agent_partial_update.sync_detailed.assert_not_called()

        MockADKAdapter.assert_called_once()

        mock_adk_adapter_instance_created = MockADKAdapter.return_value
        adapter_constructor_call_args = MockADKAdapter.call_args
        self.assertIsNotNone(adapter_constructor_call_args)
        adapter_constructor_kwargs = adapter_constructor_call_args[1]
        self.assertEqual(adapter_constructor_kwargs["id"], str(mock_created_agent_id))
        expected_adapter_config = {
            "user_id": "test_user_from_op_config",
            "name": agent_name,
            "endpoint": agent_endpoint,
        }
        self.assertEqual(adapter_constructor_kwargs["config"], expected_adapter_config)

        MockLiteLLMAdapter.assert_not_called()

        self.assertEqual(router.client, mock_client)
        self.assertIsNotNone(router.backend_agent)
        self.assertEqual(router.backend_agent.id, mock_created_agent_id)
        self.assertEqual(router.backend_agent.name, agent_name)
        expected_registry_key = str(mock_created_agent_id)
        self.assertIn(expected_registry_key, router._agent_registry)
        self.assertEqual(
            router._agent_registry[expected_registry_key],
            mock_adk_adapter_instance_created,
        )


if __name__ == "__main__":
    unittest.main()
