"""The registration helper must not silently validate an old, different config."""
import httpx
import pytest

from model_service.contracts import ModelConfig
from scripts.register_models import verify_existing_config


@pytest.mark.parametrize('changed', [False, True])
def test_existing_version_must_match_before_revalidation(changed):
    config = dict(name='model', version='1', backend='mock', task='mock',
                  capabilities=['chat'], validation_input={'messages': [{'role': 'user', 'content': 'hi'}]})
    stored = ModelConfig(**config).model_dump()
    if changed:
        stored['request_mb'] += 1
    def respond(request):
        assert request.method == 'GET' and request.url.path == '/admin/models'
        return httpx.Response(200, json={'models': [{'model_id': 'model@1', 'config': stored}]})
    with httpx.Client(base_url='http://test', transport=httpx.MockTransport(respond)) as client:
        if changed:
            with pytest.raises(SystemExit, match='new unique version'):
                verify_existing_config(client, config)
        else:
            verify_existing_config(client, config)
