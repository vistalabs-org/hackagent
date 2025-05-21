import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypeVar, Union

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.generate_request_request_messages_item import (
        GenerateRequestRequestMessagesItem,
    )


T = TypeVar("T", bound="GenerateRequestRequest")


@_attrs_define
class GenerateRequestRequest:
    """
    Attributes:
        model (Union[Unset, str]): Client-specified model (will be overridden by server)
        messages (Union[Unset, list['GenerateRequestRequestMessagesItem']]): Conversation messages
        stream (Union[Unset, bool]): Whether to stream the response Default: False.
    """

    model: Union[Unset, str] = UNSET
    messages: Union[Unset, list["GenerateRequestRequestMessagesItem"]] = UNSET
    stream: Union[Unset, bool] = False
    additional_properties: dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> dict[str, Any]:
        model = self.model

        messages: Union[Unset, list[dict[str, Any]]] = UNSET
        if not isinstance(self.messages, Unset):
            messages = []
            for messages_item_data in self.messages:
                messages_item = messages_item_data.to_dict()
                messages.append(messages_item)

        stream = self.stream

        field_dict: dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if model is not UNSET:
            field_dict["model"] = model
        if messages is not UNSET:
            field_dict["messages"] = messages
        if stream is not UNSET:
            field_dict["stream"] = stream

        return field_dict

    def to_multipart(self) -> dict[str, Any]:
        model = (
            self.model
            if isinstance(self.model, Unset)
            else (None, str(self.model).encode(), "text/plain")
        )

        messages: Union[Unset, tuple[None, bytes, str]] = UNSET
        if not isinstance(self.messages, Unset):
            _temp_messages = []
            for messages_item_data in self.messages:
                messages_item = messages_item_data.to_dict()
                _temp_messages.append(messages_item)
            messages = (None, json.dumps(_temp_messages).encode(), "application/json")

        stream = (
            self.stream
            if isinstance(self.stream, Unset)
            else (None, str(self.stream).encode(), "text/plain")
        )

        field_dict: dict[str, Any] = {}
        for prop_name, prop in self.additional_properties.items():
            field_dict[prop_name] = (None, str(prop).encode(), "text/plain")

        field_dict.update({})
        if model is not UNSET:
            field_dict["model"] = model
        if messages is not UNSET:
            field_dict["messages"] = messages
        if stream is not UNSET:
            field_dict["stream"] = stream

        return field_dict

    @classmethod
    def from_dict(cls: type[T], src_dict: Mapping[str, Any]) -> T:
        from ..models.generate_request_request_messages_item import (
            GenerateRequestRequestMessagesItem,
        )

        d = dict(src_dict)
        model = d.pop("model", UNSET)

        messages = []
        _messages = d.pop("messages", UNSET)
        for messages_item_data in _messages or []:
            messages_item = GenerateRequestRequestMessagesItem.from_dict(
                messages_item_data
            )

            messages.append(messages_item)

        stream = d.pop("stream", UNSET)

        generate_request_request = cls(
            model=model,
            messages=messages,
            stream=stream,
        )

        generate_request_request.additional_properties = d
        return generate_request_request

    @property
    def additional_keys(self) -> list[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
