import asyncio
import threading
import time
from types import SimpleNamespace
import pytest

from model_service.contracts import ModelConfig, ServiceError
from model_service.backends.ov_genai import OpenVINOGenAIBackend, validate_genai
from model_service.tasks.ov_genai import GenAITask


def config(**options):
    return ModelConfig(name='genai', version='1', task='ov_chat', backend='openvino_genai', path='models/test',
        capabilities=['chat'], validation_input={'messages':[{'role':'user','content':'hi'}]}, options=options)


@pytest.mark.parametrize('payload', [
    {'messages':[{'role':'user','content':'hi'}], 'path':'/etc/passwd'},
    {'messages':[]}, {'messages':[{'role':'user','content':'hi'}], 'max_new_tokens':9999},
])
def test_genai_rejects_invalid_or_unbounded_inputs(payload):
    with pytest.raises(ServiceError): GenAITask(config()).validate(payload, 'chat')


def test_genai_limits_provider_options_and_single_execution():
    with pytest.raises(ServiceError): validate_genai(config(CACHE_DIR='/tmp'))
    with pytest.raises(ServiceError): validate_genai(config().model_copy(update={'concurrency':2}))
    validate_genai(config(threads=4, pipeline='vlm', max_new_tokens=64))


@pytest.mark.parametrize('pipeline',[[],{},None,True,123])
def test_genai_rejects_non_string_pipeline_as_invalid_config(pipeline):
    with pytest.raises(ServiceError) as error:
        validate_genai(config(pipeline=pipeline))
    assert error.value.code=='invalid_config' and error.value.status==422


class ControlledPipeline:
    """Test double for C++ generate lifecycle, never a real model."""
    def __init__(self, *, mismatch=False):
        self.stopped=threading.Event()
        self.mismatch=mismatch
    def get_generation_config(self): return SimpleNamespace()
    def generate(self,prompt,streamer,**kwargs):
        try:
            for delta in ['你', '好', '，', '世', '界']:
                if streamer(delta)==2:
                    time.sleep(.04)
                    return 'cancelled'
                time.sleep(.01)
            return 'different' if self.mismatch else '你好，世界'
        finally:
            self.stopped.set()


def backend(pipeline):
    result=OpenVINOGenAIBackend(config(pipeline='llm'))
    result.pipeline=pipeline
    result.genai=SimpleNamespace(StreamingStatus=SimpleNamespace(CANCEL=2,RUNNING=0))
    return result


def test_genai_chunks_concatenate_and_native_stops_before_completion():
    pipeline=ControlledPipeline()
    chunks=list(backend(pipeline).stream({'prompt':'hi','max_new_tokens':8},threading.Event()))
    assert ''.join(c['delta'] for c in chunks)==chunks[-1]['text']=='你好，世界'
    assert pipeline.stopped.is_set()


def test_genai_stream_close_waits_for_native_cancel_ack():
    pipeline=ControlledPipeline();cancel=threading.Event()
    stream=backend(pipeline).stream({'prompt':'hi','max_new_tokens':8},cancel)
    assert next(stream)['delta']=='你'
    stream.close()
    assert pipeline.stopped.is_set() and cancel.is_set()


def test_genai_stream_does_not_claim_success_after_text_mismatch():
    with pytest.raises(ServiceError) as error:
        list(backend(ControlledPipeline(mismatch=True)).stream({'prompt':'hi','max_new_tokens':8},threading.Event()))
    assert error.value.code=='invalid_model_output'


def test_vision_configuration_cannot_silently_select_text_only_pipeline():
    vision = config(pipeline='llm').model_copy(update={'task': 'ov_vision_chat', 'capabilities': ['vision_chat']})
    with pytest.raises(ServiceError, match='VLM'):
        validate_genai(vision)


def test_image_placeholder_stays_inside_user_turn(monkeypatch):
    import base64
    import io
    import sys
    from PIL import Image
    capture = []
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            capture.extend(messages)
            return '<im_start>user\n' + messages[0]['content'] + '<im_end>'
        def encode(self, *args, **kwargs):
            return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 12)))
    monkeypatch.setitem(sys.modules, 'openvino_genai', SimpleNamespace(Tokenizer=lambda _: Tokenizer()))
    vision = config().model_copy(update={'task': 'ov_vision_chat', 'capabilities': ['vision_chat']})
    buffer = io.BytesIO()
    Image.new('RGB', (32, 32), 'red').save(buffer, format='PNG')
    task = GenAITask(vision)
    task.vision_metadata = {'image_processor_type':'Qwen2VLImageProcessor', 'patch_size':14, 'merge_size':2, 'min_pixels':3136, 'max_pixels':1003520}
    prepared = task.prepare({'prompt': '颜色？', 'images': [base64.b64encode(buffer.getvalue()).decode()]})
    assert capture == [{'role': 'user', 'content': '<ov_genai_image_0>\n颜色？'}]
    assert prepared.inputs['prompt'].startswith('<im_start>user\n<ov_genai_image_0>')
    assert prepared.inputs['images'][0].shape == (1, 32, 32, 3)


def test_visual_token_bound_includes_tiny_image_upscaling_and_patch_merging():
    from model_service.tasks.ov_genai import image_token_bound
    qwen25={'image_processor_type':'Qwen2VLImageProcessor','patch_size':14,'merge_size':2,'min_pixels':3136,'max_pixels':1003520}
    qwen35={'image_processor_type':'Qwen2VLImageProcessorFast','patch_size':16,'merge_size':2,'size':{'shortest_edge':65536,'longest_edge':16777216}}
    assert image_token_bound(224,224,qwen25)==66
    assert image_token_bound(32,32,qwen35)==66
    assert image_token_bound(448,448,qwen25)==258
    assert image_token_bound(4,400,qwen35)>66
    with pytest.raises(ServiceError,match='aspect'):image_token_bound(1,1000,qwen25)


def test_visual_processor_metadata_cannot_escape_bundle(tmp_path):
    outside=tmp_path/'outside.json';outside.write_text('{}')
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'preprocessor_config.json').symlink_to(outside)
    task=GenAITask(config().model_copy(update={'path':str(bundle)}))
    with pytest.raises(ServiceError) as error:task._image_token_bound(224,224)
    assert error.value.code=='path_forbidden'


def test_genai_rejects_escaped_artifact_before_constructing_native_pipeline(tmp_path, monkeypatch):
    import sys
    outside=tmp_path/'outside.bin';outside.write_bytes(b'weights')
    bundle=tmp_path/'bundle';bundle.mkdir()
    (bundle/'openvino_language_model.bin').symlink_to(outside)
    constructed=[]
    monkeypatch.setitem(sys.modules,'openvino_genai',SimpleNamespace(VLMPipeline=lambda *args,**kwargs:constructed.append(args)))
    with pytest.raises(ServiceError) as error:
        OpenVINOGenAIBackend(config().model_copy(update={'path':str(bundle)})).load()
    assert error.value.code=='path_forbidden' and constructed==[]


def test_visual_plus_text_tokens_are_checked_before_native_execution(monkeypatch):
    import base64,io,sys
    from PIL import Image
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return 'prompt'
        def encode(self,*args,**kwargs):return SimpleNamespace(input_ids=SimpleNamespace(shape=(1,12)))
    monkeypatch.setitem(sys.modules,'openvino_genai',SimpleNamespace(Tokenizer=lambda _:Tokenizer()))
    vision=config(max_input_tokens=70).model_copy(update={'task':'ov_vision_chat','capabilities':['vision_chat']})
    task=GenAITask(vision)
    task.vision_metadata={'image_processor_type':'Qwen2VLImageProcessor','patch_size':14,'merge_size':2,'min_pixels':3136,'max_pixels':1003520}
    data=io.BytesIO();Image.new('RGB',(224,224),'red').save(data,format='PNG')
    with pytest.raises(ServiceError) as error:task.prepare({'prompt':'color?','images':[base64.b64encode(data.getvalue()).decode()]})
    assert error.value.code=='input_too_large'
