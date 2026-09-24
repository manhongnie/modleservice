"""Exercise real bounded HTTP ranges and final integrity gates without external access."""
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import threading

import pytest

spec = importlib.util.spec_from_file_location('artifact_download', Path(__file__).parents[1] / 'scripts/download_models.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


@pytest.fixture
def artifact_server():
    content = b'verified weights\n' * (1024 * 1024 + 1)
    ranges = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(content)))
            self.end_headers()
        def do_GET(self):
            span = self.headers.get('Range')
            if span:
                start, end = map(int, span.removeprefix('bytes=').split('-'))
                ranges.append((start, end))
                data = content[start:end + 1]
                self.send_response(206)
                self.send_header('Content-Range', f'bytes {start}-{end}/{len(content)}')
            else:
                data = content
                self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try: yield f'http://127.0.0.1:{server.server_port}/weights', content, ranges
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize('transport', ['curl', 'httpx'])
def test_ranges_resume_and_verified_publication(artifact_server, tmp_path, monkeypatch, transport):
    import os
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT', transport)
    url, content, ranges = artifact_server
    target = tmp_path / 'weights.bin'
    partial = target.with_suffix('.bin.partial')
    with partial.open('wb') as output:
        output.truncate(len(content))
        output.write(content[:8 * 1024 * 1024])
    downloader.download(url, target, len(content), hashlib.sha256(content).hexdigest())
    assert target.read_bytes() == content and not partial.exists()
    if hasattr(os, 'SEEK_HOLE'):
        assert all(start >= 8 * 1024 * 1024 for start, _ in ranges)
    assert all(end - start < 8 * 1024 * 1024 for start, end in ranges)


@pytest.mark.parametrize('transport', ['curl', 'httpx'])
def test_digest_failure_never_publishes(artifact_server, tmp_path, monkeypatch, transport):
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT', transport)
    url, content, _ = artifact_server
    target = tmp_path / 'weights.bin'
    with pytest.raises(IOError, match='SHA256'):
        downloader.download(url, target, len(content), '0' * 64)
    assert not target.exists()


@pytest.mark.parametrize('status,headers', [
    (200, {'Content-Length': '999999999'}),
    (206, {'Content-Length': '999999999', 'Content-Range': 'bytes 0-3/999999999'}),
    (206, {'Content-Length': '4', 'Content-Range': 'bytes 4-7/100'}),
    (206, {'Content-Length': '4', 'Content-Range': 'bytes 0-3/not-a-total'}),
    (206, {'Content-Length': '4', 'Content-Range': 'bytes 0-3/3'}),
])
def test_httpx_rejects_unbounded_or_wrong_response_before_reading_body(monkeypatch, status, headers):
    import httpx
    read = []
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            read.append(True)
            yield b'never consume an unbounded response'
    def respond(request):
        return httpx.Response(status, headers=headers, stream=Body())
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT', 'httpx')
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(downloader, '_httpx_client', client)
        with pytest.raises(IOError):
            downloader.read_url('https://example.test/weights', byte_range='0-3')
    assert read == []


def test_httpx_transient_failure_keeps_known_cdn_endpoint(monkeypatch):
    import httpx
    urls=[]
    source,cdn='https://example.test/resolve/weights','https://cdn.example.test/weights'
    def respond(request):
        urls.append(str(request.url))
        if len(urls)==1:
            raise httpx.ConnectTimeout('transient failure',request=request)
        return httpx.Response(206,headers={'Content-Length':'4','Content-Range':'bytes 0-3/4'},stream=httpx.ByteStream(b'data'))
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT','httpx')
    monkeypatch.setattr(downloader,'_httpx_redirects',{source:cdn})
    monkeypatch.setattr(downloader.time,'sleep',lambda _:None)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(downloader,'_httpx_client',client)
        assert downloader.read_url(source,byte_range='0-3')==b'data'
    assert urls==[cdn,cdn]


def test_one_failed_range_preserves_other_completed_ranges_for_retry(tmp_path, monkeypatch):
    import os
    import time
    content=b'verified-range\n' * (1300 * 1024)
    attempted=[]
    fail_first=True
    def read(url, *, head=False, byte_range=None):
        if head:return url.encode()
        start,end=map(int,byte_range.split('-'))
        attempted.append(start)
        if start==0 and fail_first:
            raise IOError('Artifact transport failed: temporary connection failure')
        time.sleep(.01)
        return content[start:end+1]
    monkeypatch.setattr(downloader,'read_url',read)
    monkeypatch.setenv('MODEL_DOWNLOAD_WORKERS','1')
    monkeypatch.setenv('MODEL_DOWNLOAD_CHUNK_MB','2')
    target=tmp_path/'weights.bin'
    digest=hashlib.sha256(content).hexdigest()
    with pytest.raises(IOError,match='temporary connection'):
        downloader.download('https://example.test/weights',target,len(content),digest)
    assert not target.exists()
    partial=target.with_suffix('.bin.partial')
    with partial.open('rb') as saved:
        saved.seek(2*1024**2)
        assert saved.read()==content[2*1024**2:]
    fail_first=False
    attempted.clear()
    downloader.download('https://example.test/weights',target,len(content),digest)
    assert target.read_bytes()==content
    if hasattr(os,'SEEK_HOLE'):
        assert attempted==[0]


def test_complete_corrupt_partial_is_isolated_and_next_download_can_recover(artifact_server, tmp_path):
    url,content,ranges=artifact_server
    target=tmp_path/'weights.bin'
    partial=target.with_suffix('.bin.partial')
    corrupted=b'!'+content[1:]
    partial.write_bytes(corrupted)
    digest=hashlib.sha256(content).hexdigest()
    with pytest.raises(IOError,match='SHA256.*isolated'):
        downloader.download(url,target,len(content),digest)
    assert not target.exists() and not partial.exists()
    isolated=list(tmp_path.glob('.weights.bin.corrupt-*'))
    assert len(isolated)==1 and isolated[0].read_bytes()==corrupted
    downloader.download(url,target,len(content),digest)
    assert target.read_bytes()==content and ranges


def test_httpx_download_can_refresh_expired_signed_head_destination(tmp_path, monkeypatch):
    import httpx
    origin,old,fresh='https://origin.test/weights','https://cdn.test/old','https://cdn.test/fresh'
    size=17*1024**2
    calls=[]
    def respond(request):
        url=str(request.url);calls.append((request.method,url))
        if url==origin:
            return httpx.Response(302,headers={'Location':old if request.method=='HEAD' else fresh})
        if request.method=='HEAD':
            return httpx.Response(200,headers={'Content-Length':str(size)})
        if url==old:return httpx.Response(403)
        start,end=map(int,request.headers['Range'].removeprefix('bytes=').split('-'))
        return httpx.Response(206,headers={'Content-Length':str(end-start+1),'Content-Range':f'bytes {start}-{end}/{size}'},
                              stream=httpx.ByteStream(b'x'*(end-start+1)))
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT','httpx')
    monkeypatch.setenv('MODEL_DOWNLOAD_WORKERS','2')
    monkeypatch.setattr(downloader,'_httpx_redirects',{})
    monkeypatch.setattr(downloader.time,'sleep',lambda _:None)
    with httpx.Client(transport=httpx.MockTransport(respond),follow_redirects=True) as client:
        monkeypatch.setattr(downloader,'_httpx_client',client)
        target=tmp_path/'weights.bin'
        downloader.download(origin,target,size,hashlib.sha256(b'x'*size).hexdigest())
        assert target.stat().st_size==size
    assert calls.count(('GET',old))==1 and calls.count(('GET',origin))==1
    assert downloader._httpx_redirects[origin]==fresh


@pytest.mark.parametrize('late_status',[206,403])
def test_old_response_cannot_overwrite_or_remove_newer_cdn_cache(monkeypatch, late_status):
    import httpx
    origin,old,fresh='https://origin.test/weights','https://cdn.test/old','https://cdn.test/fresh'
    def respond(request):
        if str(request.url)==old:
            # Another worker has refreshed the cache while this old request was in flight.
            downloader._httpx_redirects[origin]=fresh
            if late_status==403:return httpx.Response(403)
        assert str(request.url) in {old,fresh}
        return httpx.Response(206,headers={'Content-Length':'4','Content-Range':'bytes 0-3/4'},stream=httpx.ByteStream(b'data'))
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT','httpx')
    monkeypatch.setattr(downloader,'_httpx_redirects',{origin:old})
    monkeypatch.setattr(downloader.time,'sleep',lambda _:None)
    with httpx.Client(transport=httpx.MockTransport(respond),follow_redirects=True) as client:
        monkeypatch.setattr(downloader,'_httpx_client',client)
        assert downloader.read_url(origin,byte_range='0-3')==b'data'
    assert downloader._httpx_redirects[origin]==fresh


def test_deadline_checks_each_received_fragment_without_filling_a_64k_buffer(monkeypatch):
    import httpx
    now=[0]
    past_deadline=[]
    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            yield b'a'
            now[0]+=121
            yield b'b'
            past_deadline.append(True)
            yield b'cd'
    def respond(request):
        return httpx.Response(206,headers={'Content-Length':'4','Content-Range':'bytes 0-3/4'},stream=Drip())
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT','httpx')
    monkeypatch.setattr(downloader.time,'monotonic',lambda:now[0])
    monkeypatch.setattr(downloader.time,'sleep',lambda _:None)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(downloader,'_httpx_client',client)
        with pytest.raises(IOError,match='transport failed'):
            downloader.read_url('https://origin.test/weights',byte_range='0-3')
    assert past_deadline==[]


@pytest.mark.parametrize('advertise',[False,True])
def test_known_small_file_size_bounds_response_before_publication(tmp_path, monkeypatch, advertise):
    import httpx
    consumed=[]
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            consumed.append(True)
            yield b'oversized'
    def respond(request):
        return httpx.Response(200,headers={'Content-Length':'9'} if advertise else {},stream=Body())
    monkeypatch.setenv('MODEL_DOWNLOAD_TRANSPORT','httpx')
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(downloader,'_httpx_client',client)
        target=tmp_path/'metadata.json'
        with pytest.raises(IOError,match='bounded'):
            downloader.download('https://origin.test/metadata',target,expected_size=4)
    assert not target.exists()
    assert consumed==([] if advertise else [True])
