#!/usr/bin/env python3
"""Verify real Qwen token streaming, cancellation, reuse and process exit.

Run from the repository root:
    PYTHONPATH=. .venv/bin/python scripts/validate_real_streaming.py
This checks the process execution boundary, not HTTP transport or ASR/TTS quality.
The Qwen files listed in examples/models.real.json must already be available.
"""
import asyncio
import json
import time
from pathlib import Path
from model_service.contracts import ModelConfig, ServiceError
from model_service.executor import ProcessExecutor

async def main():
    cfg=ModelConfig(**json.loads(Path('examples/models.real.json').read_text())[0])
    executor=ProcessExecutor(load_timeout_s=180)
    report={'model':cfg.model_id,'test':'real_qwen_incremental_stream', 'boundary':'ProcessExecutor (not HTTP)', 'timestamp_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    try:
        report['load']=await executor.load(cfg)
        prompt={'messages':[{'role':'user','content':'请用中文介绍什么是机器学习。'}],'max_new_tokens':64}
        chunks=[]
        started=time.monotonic()
        async for chunk in executor.stream(cfg.instance_key,prompt,'normal',True,asyncio.Event()):
            chunks.append({'at_seconds':round(time.monotonic()-started,3),**chunk})
        report['chunks']=chunks
        assert len(chunks)>2
        assert chunks[0]['at_seconds'] < chunks[-1]['at_seconds']
        assert ''.join(chunk['delta'] for chunk in chunks)==chunks[-1]['text']
        assert all('\ufffd' not in chunk['delta'] for chunk in chunks)
        assert chunks[-1]['finish_reason'] in {'stop', 'length'}
        cancel=asyncio.Event()
        stream=executor.stream(cfg.instance_key,prompt,'cancelled',True,cancel)
        first=await anext(stream)
        started=time.monotonic()
        await stream.aclose()
        report['cancel']={'first_chunk':first,'stop_seconds':round(time.monotonic()-started,3),'signalled':cancel.is_set(), 'state':executor.snapshot()[cfg.instance_key]}
        assert cancel.is_set() and report['cancel']['state']['busy']==0
        reuse=[chunk async for chunk in executor.stream(cfg.instance_key,cfg.validation_input,'reuse',False,asyncio.Event())]
        report['reuse']=reuse
        assert report['cancel']['state']['load_count']==1
        report['unload']=await executor.unload(cfg.instance_key)
        report['status']='passed'
    finally:
        await executor.close()
        Path('docs/real-streaming-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({key: value for key,value in report.items() if key!='chunks'},ensure_ascii=False))

if __name__=='__main__':
    asyncio.run(main())
