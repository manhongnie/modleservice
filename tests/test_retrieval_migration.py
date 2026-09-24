import math
from threading import Event

import numpy as np
import pytest

from model_service.contracts import ModelConfig, ServiceError
from model_service.tasks.retrieval_ov import BM42Task, DualClipTask
from model_service.backends.retrieval_ov import DualClipOpenVINOBackend


def config(task='bm42_local', **changes):
    values = dict(name='retrieval', version='test', task=task, backend='openvino',
                  capabilities=['sparse_embeddings'], validation_input={'texts':['hello']})
    values.update(changes)
    return ModelConfig(**values)


class Tokenizer:
    all_special_tokens = ['[CLS]', '[SEP]', '[PAD]']
    vocab = ['[CLS]', 'the', 'play', '##ing', 'play', '.', '[SEP]']
    def convert_ids_to_tokens(self, ids): return [self.vocab[i] for i in ids]


class Stemmer:
    def stem(self, word): return 'play' if word == 'playing' else word


def bm42_task():
    task = BM42Task(config())
    task.tokenizer = Tokenizer()
    task._lexical = ({'the'}, Stemmer())
    return task


def test_bm42_real_attention_weighting_merges_subwords_and_repeated_stems():
    import mmh3
    task = bm42_task()
    attention = np.zeros((1, 2, 7, 7), dtype=np.float32)
    attention[:, :, 0, :] = [0, .7, .02, .03, .1, .15, 0]
    context = {'input_ids': np.arange(7)[None], 'attention_mask': np.ones((1,7)), 'input_type':'document'}
    result = task.finish({'attention_6': attention}, context)
    assert result['requires_idf'] is True
    assert result['sparse_embeddings'][0]['indices'] == [abs(mmh3.hash('play'))]
    assert result['sparse_embeddings'][0]['values'] == pytest.approx([math.sqrt(math.log1p(.1))])
    attention[:, :, 0, 2:4] = [.2, .3]
    changed = task.finish({'attention_6':attention}, context)
    assert changed['sparse_embeddings'][0]['values'] == pytest.approx([math.sqrt(math.log1p(.5))])
    context['input_type'] = 'query'
    assert task.finish({'attention_6':attention},context)['sparse_embeddings'][0]['values'] == [1.0]


@pytest.mark.parametrize('attention', [np.zeros((1,7,7)), np.full((1,2,7,7), np.nan), np.full((1,2,7,7), -.2)])
def test_bm42_rejects_malformed_or_negative_attention(attention):
    task = bm42_task()
    with pytest.raises(ServiceError):
        task.finish({'attention_6':attention}, {'input_ids':np.arange(7)[None], 'attention_mask':np.ones((1,7)), 'input_type':'document'})


@pytest.mark.parametrize('payload', [{'texts':['x'],'input_type':'arbitrary'}, {'texts':[]}, {'texts':['x'],'path':'/etc/passwd'}])
def test_bm42_input_contract(payload):
    with pytest.raises(ServiceError): bm42_task().validate(payload,'sparse_embeddings')


def test_dual_clip_single_modality_and_combined_outputs():
    task = DualClipTask(config(task='dual_clip'))
    assert task.finish({'text_embeds':np.array([[3,4]])},{'text':1})['embeddings'][0] == pytest.approx([.6,.8])
    result = task.finish({'text_embeds':np.array([[3,4]]),'image_embeds':np.array([[0,4]])},{'text':1,'image':1})
    assert result['text_embeddings'][0] == pytest.approx([.6,.8])
    assert result['image_embeddings'][0] == pytest.approx([0.,1.])
    with pytest.raises(ServiceError): task.finish({'text_embeds':np.array([[3,4]])},{'text':2})


def test_dual_clip_load_failure_releases_already_loaded_tower(monkeypatch):
    import model_service.backends.retrieval_ov as backend_module
    created=[]
    class Backend:
        def __init__(self,cfg): self.cfg=cfg; self.closed=False;created.append(self)
        def load(self):
            if self.cfg.options['model_file']=='openvino_image.xml': raise ServiceError('load_failed','bad image graph')
            return {}
        def close(self): self.closed=True
    monkeypatch.setattr(backend_module,'OpenVINOBackend',Backend)
    backend=DualClipOpenVINOBackend(config(task='dual_clip',path='models/clip'))
    with pytest.raises(ServiceError): backend.load()
    assert len(created)==2 and all(x.closed for x in created)
    assert backend.towers=={}


def test_dual_clip_uses_requested_tower_only_and_propagates_cancellation():
    calls=[]
    class Tower:
        def infer(self,values,cancel):
            calls.append(values)
            if cancel.is_set(): raise ServiceError('cancelled','confirmed',499)
            return {'text_embeds': np.array([[1,0]])}
    backend=DualClipOpenVINOBackend(config(task='dual_clip'))
    backend.towers={'text':Tower(),'image':Tower()}
    backend.infer({'text':{'input_ids':[[1]]}},Event())
    assert len(calls)==1
    cancel=Event();cancel.set()
    with pytest.raises(ServiceError,match='confirmed'):backend.infer({'text':{}},cancel)


def test_qwen_embedding_pools_last_nonpadding_token_on_both_sides():
    from model_service.tasks.retrieval_ov import QwenEmbeddingTask
    task=QwenEmbeddingTask(config(task='qwen3_embedding_ov',capabilities=['embeddings']))
    outputs={'last_hidden_state':np.array([[[0,0],[3,4],[9,0]], [[9,0],[0,2],[3,4]]])}
    result=task.finish(outputs,{'attention_mask':np.array([[1,1,0],[0,1,1]])})
    assert result['embeddings'][0]==pytest.approx([.6,.8])
    assert result['embeddings'][1]==pytest.approx([.6,.8])
    with pytest.raises(ServiceError): task.finish(outputs,{'attention_mask':np.zeros((2,3))})


def test_qwen_reranker_yes_no_probability_uses_terminal_token():
    from model_service.tasks.retrieval_ov import QwenRerankerTask
    task=QwenRerankerTask(config(task='qwen3_reranker_ov',capabilities=['rerank']))
    logits=np.zeros((2,3,6))
    logits[0,-1,1:3]=[1000,1002]
    logits[1,-1,1:3]=[1000,998]
    result=task.finish({'logits':logits},{'count':2,'no_id':1,'yes_id':2})
    assert result['results'][0]['index']==0
    assert result['results'][0]['score']==pytest.approx(1/(1+math.exp(-2)))


def test_imported_dense_batch_budget_is_enforced():
    from model_service.tasks.retrieval_ov import BoundedDenseTask,BoundedRerankTask
    with pytest.raises(ServiceError):
        BoundedDenseTask(config(task='bge_m3_bounded')).validate({'texts':['x']*9},'embeddings')
    with pytest.raises(ServiceError):
        BoundedRerankTask(config(task='bge_reranker_bounded')).validate({'query':'x','documents':['x']*9},'rerank')


def test_retrieval_catalog_validation_keeps_control_process_lightweight():
    import subprocess
    import sys
    script='''
import sys,json
from model_service.contracts import ModelConfig,Settings
from model_service.plugins import BuiltinPluginCatalog
catalog=BuiltinPluginCatalog();settings=Settings(business_keys=['b'],admin_keys=['a'])
for row in json.load(open('examples/models.retrieval-migrated.json')):
 config=ModelConfig(**row);catalog.validate(config,settings)
 for capability in config.capabilities:catalog.validate_input(config,config.validation_input,capability)
assert not {'torch','transformers','numpy','openvino','nltk'}.intersection(sys.modules)
'''
    subprocess.run([sys.executable,'-c',script],check=True)


@pytest.mark.parametrize('options', [{'max_length':True},{'max_batch_size':0},{'max_length':513}])
def test_bm42_rejects_invalid_model_bounds_before_loading(options):
    task=BM42Task(config(options=options))
    with pytest.raises(ServiceError) as error: task.validate({'texts':['hello']},'sparse_embeddings')
    assert error.value.code=='invalid_config'


def test_dual_clip_backend_compiles_and_runs_both_native_openvino_graphs(tmp_path):
    ov=pytest.importorskip('openvino')
    from openvino import opset13 as ops
    for tower, input_name in [('text','input_ids'),('image','pixel_values')]:
        values=ops.parameter([-1,2],np.float32,name=input_name)
        projected=ops.multiply(values,ops.constant(np.array([2,3],dtype=np.float32)))
        projected.output(0).get_tensor().set_names({f'{tower}_embeds'})
        graph=ov.Model([projected],[values])
        ov.save_model(graph,tmp_path/f'openvino_{tower}.xml')
    cfg=config(task='dual_clip',backend='openvino_clip',path=str(tmp_path),capabilities=['embeddings','image_embeddings'])
    backend=DualClipOpenVINOBackend(cfg)
    metadata=backend.load()
    try:
        assert set(metadata['towers'])=={'text','image'}
        output=backend.infer({'text':{'input_ids':[[1,2]]},'image':{'pixel_values':[[3,4]]}},Event())
        assert output['text_embeds'].tolist()==[[2,6]]
        assert output['image_embeds'].tolist()==[[6,12]]
        cancel=Event();cancel.set()
        with pytest.raises(ServiceError) as error:backend.infer({'text':{'input_ids':[[1,2]]}},cancel)
        assert error.value.code=='cancelled'
    finally:assert backend.close()['unloaded'] is True


@pytest.mark.parametrize('value',[[],{},False,None])
def test_bm42_invalid_input_type_is_a_client_error(value):
    with pytest.raises(ServiceError) as error:bm42_task().validate({'texts':['hello'],'input_type':value},'sparse_embeddings')
    assert error.value.code=='invalid_input'
