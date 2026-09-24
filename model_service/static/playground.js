"use strict";

// The playground uses only the public HTTP contract. No model files or admin APIs.
(() => {
  const $ = (id) => document.getElementById(id);
  const encoder = new TextEncoder();
  const state = {
    key: "", models: [], model: null, capability: "", mode: "form",
    maxBodyBytes: 0, controller: null, busy: false, generation: 0,
    files: new Map(), readers: new Set(), urls: [], output: {},
    raw: null, events: [], lastRequest: null, started: 0,
  };
  const textField = (key, label, sample, hint = "") => ({key, label, sample, hint, type: "text"});
  const numberField = (key, label, sample, min, max, step = 1) => ({key, label, sample, min, max, step, type: "number"});
  const fileField = (key, label, kind, optional = false) => ({key, label, kind, optional, type: "file"});
  const tokenField = numberField("max_new_tokens", "最大生成 Token", 128, 1, 256);
  const imageFile = fileField("images", "上传图片", "image");
  const audioFile = fileField("audio_base64", "上传音频", "audio");
  const promptField = textField("prompt", "画面描述", "A small cabin by a peaceful lake, soft morning light, watercolor painting.", "英文描述通常更适合图像与视频生成模型。");
  const imageParameters = [numberField("width", "宽度 / px", 512, 128, 512, 64), numberField("height", "高度 / px", 512, 128, 512, 64), numberField("steps", "生成步数", 1, 1, 4), numberField("seed", "随机种子", 42, 0, 2147483647)];
  const schemas = {
    chat: {name: "文本对话", symbol: "↳", fields: [textField("messages", "你的问题", "用三句话介绍一下你自己。", "默认发送一条用户消息；多轮对话可使用 JSON 模式。"), tokenField]},
    vision_chat: {name: "看图问答", symbol: "▧", fields: [textField("prompt", "想了解图片的什么？", "请描述图片中的主要内容。"), imageFile, tokenField]},
    embeddings: {name: "文本向量", symbol: "≋", fields: [textField("texts", "待编码文本", "北京是中国的首都\n中国的首都是北京\n今天的天气很好", "每行一段文本，每段生成一个向量。") ]},
    image_embeddings: {name: "图像向量", symbol: "▦", fields: [imageFile]},
    sparse_embeddings: {name: "稀疏向量", symbol: "⁙", fields: [textField("texts", "待编码文本", "A quiet library full of books\nReading opens new worlds", "每行一段文本；BM42 适合英文，检索时需结合语料 IDF。") ]},
    rerank: {name: "文本重排序", symbol: "≡", fields: [textField("query", "检索问题", "中国的首都是哪里？"), textField("documents", "候选文档", "上海是中国的经济中心。\n北京是中华人民共和国的首都。\n杭州以西湖闻名。", "每行一篇文档，结果按相关性从高到低排列。") ]},
    asr: {name: "语音识别", symbol: "〰", fields: [audioFile]},
    tts: {name: "文字转语音", symbol: "♫", fields: [textField("text", "想说的话", "你好，欢迎使用模型服务。让我们从一个简单的想法开始。"), numberField("speed", "语速", 1, 0.5, 2, 0.1)]},
    speaker_embeddings: {name: "声纹识别", symbol: "◉", fields: [audioFile, fileField("compare_audio_base64", "对比音频（可选）", "audio", true)]},
    voice_clone: {name: "声音克隆", symbol: "♬", fields: [textField("text", "要生成的语音文本", "你好，很高兴在这里与你见面。"), fileField("reference_audio_base64", "参考音频", "audio"), textField("reference_text", "参考音频原文", "", "准确填写参考音频中说出的文字。"), numberField("num_steps", "生成步数", 4, 1, 8)]},
    image_generation: {name: "图片生成", symbol: "▧", fields: [promptField, ...imageParameters]},
    video_generation: {name: "视频生成", symbol: "▷", fields: [promptField, ...imageParameters.map((field) => ({...field, sample: field.key === "width" || field.key === "height" ? 256 : field.key === "steps" ? 4 : field.sample})), numberField("frames", "帧数", 8, 4, 16), numberField("fps", "帧率 / fps", 8, 1, 16)]},
  };

  function node(tag, className, value) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== undefined) element.textContent = String(value);
    return element;
  }
  function notice(message = "") {
    $("notice").textContent = message;
    $("notice").hidden = !message;
  }
  function connectionStatus(message, tone = "idle") {
    $("connection-status").textContent = message;
    $("connection-status").dataset.tone = tone;
  }
  function resultStatus(message, tone) {
    $("result-status").textContent = message;
    $("result-status").dataset.tone = tone;
    $("result-status").classList.toggle("loading-text", tone === "running");
    document.querySelector(".result-panel").setAttribute("aria-busy", String(tone === "running"));
  }
  function resultMessage(message = "", tone = "idle") {
    $("result-message").textContent = message;
    $("result-message").dataset.tone = tone;
    $("result-message").hidden = !message;
  }
  function releaseMedia() {
    state.urls.forEach((url) => URL.revokeObjectURL(url));
    state.urls = [];
  }
  function resetResult() {
    releaseMedia();
    state.output = {};
    state.raw = null;
    state.events = [];
    state.lastRequest = null;
    $("result-content").replaceChildren();
    $("result-content").hidden = true;
    $("result-empty").hidden = false;
    $("result-meta").hidden = true;
    $("result-mock").hidden = true;
    $("response-details").hidden = true;
    $("response-details").open = false;
    $("response-json").textContent = "";
    resultMessage();
    resultStatus("等待输入", "idle");
  }
  function resetFiles() {
    state.readers.forEach((reader) => reader.abort());
    state.readers.clear();
    state.files.clear();
  }
  function refreshControls() {
    const locked = state.busy || state.readers.size > 0;
    document.querySelectorAll("[data-edit]").forEach((element) => {element.disabled = locked;});
    const unavailable = !state.model || !schemas[state.capability];
    $("model-select").disabled = locked || !state.models.length;
    ["example-button", "form-mode", "json-mode"].forEach((id) => {$(id).disabled = locked || unavailable;});
    $("run-button").disabled = state.busy || unavailable || state.readers.size > 0;
    $("cancel-button").hidden = !state.busy || !state.lastRequest;
    $("run-label").textContent = state.busy ? "正在运行…" : "运行测试";
    $("connect-button").disabled = locked;
    $("clear-key").disabled = !state.key && !$("api-key").value && !state.busy;
  }
  function clearDirectory() {
    state.models = [];
    state.model = null;
    state.capability = "";
    state.maxBodyBytes = 0;
    $("model-select").replaceChildren(new Option("先连接服务", ""));
    $("model-count").textContent = "0";
    $("model-id").textContent = "可用模型将在这里显示";
    $("model-mock").hidden = true;
    $("capabilities").replaceChildren(node("p", "subtle-text", "连接后选择要测试的能力。"));
    $("inference-form").hidden = true;
    $("input-placeholder").hidden = false;
    $("form-fields").replaceChildren();
    $("json-input").value = "";
    $("request-json").textContent = "";
    $("endpoint").textContent = "POST /v1/…";
  }
  function disconnect() {
    const wasRunning = state.busy && Boolean(state.lastRequest);
    state.generation += 1;
    if (state.controller) state.controller.abort();
    state.controller = null;
    state.busy = false;
    state.key = "";
    $("api-key").value = "";
    resetFiles();
    clearDirectory();
    resetResult();
    notice(wasRunning ? "已清除密钥并停止接收结果；服务仍会等待底层任务清理。" : "");
    connectionStatus("密钥已清除，尚未连接");
    $("connection-description").textContent = "输入业务密钥，读取可用模型";
    refreshControls();
  }
  function apiError(body, status) {
    const error = body && body.error;
    if (error && typeof error.message === "string") return new Error(`${error.code || "request_failed"} · ${error.message}`);
    const hint = status === 401 || status === 403 ? "请检查业务 API 密钥。" : status === 404 ? "接口不存在，请确认运行的是支持网页测试的服务版本。" : "服务未返回有效结果，请检查服务日志。";
    return new Error(`HTTP ${status} · ${hint}`);
  }
  async function readJSON(response) {
    const value = await response.text();
    try {return JSON.parse(value);} catch {throw new Error(`HTTP ${response.status} · 服务未返回有效 JSON，请检查服务或反向代理配置。`);}
  }
  async function connect(event) {
    event.preventDefault();
    if (state.busy) return;
    const key = $("api-key").value.trim();
    if (!key) {notice("请先填写 BUSINESS_API_KEY 业务密钥。"); $("api-key").focus(); return;}
    state.generation += 1;
    const generation = state.generation;
    resetFiles(); clearDirectory(); resetResult(); notice();
    state.key = "";
    state.busy = true;
    const controller = new AbortController();
    state.controller = controller;
    connectionStatus("正在连接…"); refreshControls();
    try {
      const response = await fetch("/v1/models", {headers: {Authorization: `Bearer ${key}`}, credentials: "omit", cache: "no-store", signal: controller.signal});
      const data = await readJSON(response);
      if (!response.ok) throw apiError(data, response.status);
      if (!data || !Array.isArray(data.models)) throw new Error("模型目录格式无效，请检查服务版本。");
      if (generation !== state.generation) return;
      state.key = key;
      state.models = data.models.filter((model) => model && typeof model.model_id === "string" && Array.isArray(model.capabilities));
      state.maxBodyBytes = Number(data.max_body_bytes) || 0;
      $("model-count").textContent = state.models.length;
      $("model-select").replaceChildren(...state.models.map((model) => new Option(`${model.name || model.model_id}${model.mock ? " · MOCK" : ""}`, model.model_id)));
      connectionStatus(`已连接 · ${state.models.length} 个可用模型`, "success");
      $("connection-description").textContent = "业务密钥已验证，可开始测试";
      state.busy = false;
      state.controller = null;
      if (state.models.length) selectModel();
      else {
        $("model-select").append(new Option("暂无可用模型", ""));
        $("model-id").textContent = "目录为空";
        notice("当前没有可调用模型。请管理员先准备模型文件、登记并启用模型，再点击「连接服务」刷新目录。");
      }
    } catch (error) {
      if (generation !== state.generation) return;
      if (error.name !== "AbortError") {notice(error.message || "连接失败，请检查网络和服务。"); connectionStatus("连接失败", "error");}
    } finally {
      if (generation === state.generation) {state.busy = false; state.controller = null; refreshControls();}
    }
  }
  function selectModel() {
    state.model = state.models.find((model) => model.model_id === $("model-select").value) || null;
    if (!state.model) return;
    $("model-id").textContent = state.model.model_id;
    $("model-mock").hidden = !state.model.mock;
    const buttons = state.model.capabilities.filter((capability) => schemas[capability]).map((capability) => {
      const button = node("button", "capability-button");
      button.type = "button";
      button.dataset.edit = "";
      button.dataset.capability = capability;
      button.append(node("span", "capability-symbol", schemas[capability].symbol), node("span", "capability-title", schemas[capability].name), node("span", "capability-arrow", "›"));
      button.addEventListener("click", () => {if (!state.busy) selectCapability(capability);});
      return button;
    });
    $("capabilities").replaceChildren(...buttons);
    selectCapability(buttons.length ? buttons[0].dataset.capability : "");
    if (!buttons.length) notice("这个模型尚无网页支持的能力，请使用其 HTTP API。");
  }
  function fieldOptions(field) {
    const limits = state.model.limits || {};
    const result = {...field};
    if (field.type === "number") {
      const maximum = limits[field.key === "num_steps" ? "max_steps" : field.key === "max_new_tokens" ? "max_new_tokens" : `max_${field.key}`];
      if (Number.isFinite(maximum)) result.max = Math.min(result.max, maximum);
      const fixed = limits[`fixed_${field.key}`];
      const preset = limits[`default_${field.key}`];
      if (Number.isFinite(preset)) result.sample = preset;
      if (Number.isFinite(fixed)) result.min = result.max = result.sample = fixed;
      result.sample = Math.max(result.min, Math.min(result.sample, result.max));
      if (state.capability === "video_generation" && field.key === "steps") result.min = result.max = result.sample = 4;
    }
    if (field.type === "text") {
      const max = field.key === "reference_text" ? 300 : field.key === "prompt" ? limits.max_prompt_chars || limits.max_text_chars : limits.max_text_chars;
      if (Number.isFinite(max)) result.maxLength = max;
    }
    return result;
  }
  function selectCapability(capability) {
    state.generation += 1;
    resetFiles(); resetResult(); notice();
    state.capability = capability;
    state.mode = "form";
    $("form-fields").replaceChildren();
    $("json-input").value = "";
    $("request-json").textContent = "";
    $("request-details").open = false;
    $("inference-form").hidden = !schemas[capability];
    $("input-placeholder").hidden = Boolean(schemas[capability]);
    document.querySelectorAll(".capability-button").forEach((button) => {
      const selected = button.dataset.capability === capability;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    if (!schemas[capability]) {refreshControls(); return;}
    $("endpoint").textContent = `POST /v1/${capability}`;
    schemas[capability].fields.forEach((original) => {
      const field = fieldOptions(original);
      const wrapper = node("div", `field${field.type === "number" ? "" : " field-wide"}`);
      const label = node("label", "field-label", field.label);
      label.htmlFor = `field-${field.key}`;
      const input = node(field.type === "text" ? "textarea" : "input");
      input.id = `field-${field.key}`;
      input.name = field.key;
      input.dataset.edit = "";
      input.required = !field.optional;
      const hint = node("p", "field-hint", field.hint || "");
      hint.id = `hint-${field.key}`;
      input.setAttribute("aria-describedby", hint.id);
      if (field.type === "text") {
        input.rows = field.key === "query" || field.key === "reference_text" ? 3 : 5;
        input.value = field.sample;
        if (field.maxLength) {input.maxLength = field.maxLength; input.value = input.value.slice(0, field.maxLength);}
        if (!field.hint && field.maxLength) hint.textContent = `最多 ${field.maxLength} 个字符。`;
      } else if (field.type === "number") {
        input.type = "number"; input.min = field.min; input.max = field.max; input.step = field.step; input.value = field.sample;
        hint.textContent = field.min === field.max ? `此模型固定为 ${field.sample}` : `${field.min} – ${field.max}`;
      } else {
        input.type = "file";
        input.accept = field.kind === "image" ? "image/png,image/jpeg,image/webp,.png,.jpg,.jpeg,.webp" : ".wav,audio/wav,audio/x-wav,audio/wave";
        input.multiple = field.kind === "image" && (state.capability !== "vision_chat" || (state.model.limits || {}).max_images > 1);
        hint.textContent = field.kind === "image" ? "PNG、JPG 或 WebP；发送时自动转为 Base64。" : `单声道、16 kHz、PCM16 WAV${field.key === "reference_audio_base64" ? "，至少 1 秒" : ""}。`;
        const audioLimit = (state.model.limits || {})[field.key === "reference_audio_base64" ? "max_reference_seconds" : "max_audio_seconds"];
        if (field.kind === "audio" && audioLimit) hint.textContent += ` 最长 ${audioLimit} 秒。`;
        input.addEventListener("change", () => readFiles(field, input, hint));
      }
      input.addEventListener("input", updateRequestPreview);
      wrapper.append(label, input);
      if (hint.textContent) wrapper.append(hint);
      $("form-fields").append(wrapper);
    });
    $("stream-wrap").hidden = !["chat", "vision_chat"].includes(capability);
    $("stream").checked = true;
    setMode("form");
    refreshControls();
  }
  function readFile(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      state.readers.add(reader); refreshControls();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
      reader.onerror = () => reject(new Error(`无法读取文件：${file.name}`));
      reader.onabort = () => reject(new DOMException("文件读取已取消", "AbortError"));
      reader.onloadend = () => {state.readers.delete(reader); refreshControls();};
      reader.readAsDataURL(file);
    });
  }
  async function readFiles(field, input, hint) {
    const generation = state.generation;
    const files = Array.from(input.files || []);
    state.files.delete(field.key); notice();
    if (!files.length) {hint.textContent = "尚未选择文件。"; updateRequestPreview(); return;}
    try {
      const limits = state.model.limits || {};
      const countLimit = field.kind === "image" ? Math.min(limits.max_images || (state.capability === "vision_chat" ? 1 : 16), limits.max_batch_size || limits.max_batch || 16) : 1;
      if (files.length > countLimit) throw new Error(`最多选择 ${countLimit} 个文件。`);
      const maxBytes = Math.min(...[limits.max_input_bytes, state.maxBodyBytes].filter((value) => value > 0), 64 * 1024 * 1024);
      let size = 0;
      for (const file of files) {
        const valid = field.kind === "image" ? /\.(png|jpe?g|webp)$/i.test(file.name) && (!file.type || /^image\/(png|jpeg|webp)$/.test(file.type)) : /\.wav$/i.test(file.name) && (!file.type || ["audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"].includes(file.type));
        if (!valid) throw new Error(field.kind === "image" ? "请选择 PNG、JPG 或 WebP 图片。" : "请选择 WAV 音频；不支持直接上传 MP3、M4A。" );
        if (!file.size) throw new Error(`文件为空：${file.name}`);
        size += Math.ceil(file.size / 3) * 4;
      }
      if (size >= maxBytes) throw new Error(`文件编码后超过请求限制（${formatBytes(maxBytes)}），请缩小文件。`);
      hint.textContent = "正在读取文件…";
      // Read sequentially to avoid several large decoded files in memory at once.
      const values = [];
      for (const file of files) values.push(await readFile(file));
      if (generation !== state.generation) return;
      state.files.set(field.key, field.kind === "image" ? values : values[0]);
      hint.textContent = files.map((file) => `${file.name} · ${formatBytes(file.size)}`).join(" / ");
      hint.classList.add("file-status");
    } catch (error) {
      if (generation !== state.generation || error.name === "AbortError") return;
      input.value = "";
      hint.textContent = "请重新选择文件。";
      hint.classList.remove("file-status");
      notice(error.message);
    } finally {
      if (generation === state.generation) {refreshControls(); updateRequestPreview();}
    }
  }
  function formatBytes(bytes) {
    return bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.ceil(bytes / 1024)} KB`;
  }
  function formInput(validate) {
    const value = {};
    for (const original of schemas[state.capability].fields) {
      const field = fieldOptions(original);
      const input = $(`field-${field.key}`);
      if (field.type === "file") {
        const file = state.files.get(field.key);
        if (!file && !field.optional && validate) throw new Error(`请先${field.label}。`);
        if (file) value[field.key] = file;
      } else if (field.type === "number") {
        const number = Number(input.value);
        if (validate && (!input.value.trim() || !Number.isFinite(number) || !input.validity.valid)) throw new Error(`「${field.label}」须为 ${field.min} 到 ${field.max} 之间的有效数值。`);
        value[field.key] = number;
      } else {
        const text = input.value.trim();
        if (validate && !text) throw new Error(`请填写「${field.label}」。`);
        if (validate && field.maxLength && [...text].length > field.maxLength) throw new Error(`「${field.label}」最多 ${field.maxLength} 个字符。`);
        if (field.key === "messages") value.messages = [{role: "user", content: text}];
        else if (["texts", "documents"].includes(field.key)) {
          value[field.key] = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
          const maxBatch = (state.model.limits || {}).max_batch_size || (state.model.limits || {}).max_batch || 128;
          if (validate && value[field.key].length > maxBatch) throw new Error(`每次最多输入 ${maxBatch} 段文本。`);
        } else value[field.key] = text;
      }
    }
    return value;
  }
  function requestBody(validate = false) {
    let input;
    if (state.mode === "json") {
      try {input = JSON.parse($("json-input").value);} catch {throw new Error("JSON 格式不正确，请检查引号、逗号和括号。");}
      if (!input || typeof input !== "object" || Array.isArray(input)) throw new Error("input 必须是 JSON 对象。");
    } else input = formInput(validate);
    return {model: state.model.model_id, input, stream: !$("stream-wrap").hidden && $("stream").checked};
  }
  function setMode(mode) {
    if (state.busy || !state.model) return;
    if (mode === "json" && state.mode !== "json") $("json-input").value = JSON.stringify(formInput(false), null, 2);
    state.mode = mode;
    $("form-fields").hidden = mode === "json";
    $("json-input-wrap").hidden = mode !== "json";
    $("form-mode").classList.toggle("active", mode === "form");
    $("json-mode").classList.toggle("active", mode === "json");
    $("form-mode").setAttribute("aria-pressed", String(mode === "form"));
    $("json-mode").setAttribute("aria-pressed", String(mode === "json"));
    $("json-hint").textContent = "只填写 input 对象。JSON 修改仅在本模式生效；返回表单会恢复原表单内容。";
    updateRequestPreview();
  }
  function updateRequestPreview() {
    if (!$("request-details").open || !state.model) return;
    try {$("request-json").textContent = JSON.stringify(requestBody(), null, 2);} catch (error) {$("request-json").textContent = error.message;}
  }
  function fillExample() {
    if (state.busy || !schemas[state.capability]) return;
    schemas[state.capability].fields.forEach((original) => {
      const field = fieldOptions(original);
      if (field.type !== "file") $(`field-${field.key}`).value = field.maxLength ? String(field.sample).slice(0, field.maxLength) : field.sample;
    });
    if (state.mode === "json") $("json-input").value = JSON.stringify(formInput(false), null, 2);
    notice(schemas[state.capability].fields.some((field) => field.type === "file" && !field.optional && !state.files.has(field.key)) ? "文字示例已填入，请选择自己的图片或 WAV 音频；文件不会自动上传。" : "");
    updateRequestPreview();
  }
  // Match the service's UTF-8 JSON input accounting (spaces after separators).
  function inputBytes(value) {
    const serialized = JSON.stringify(value);
    let quoted = false, escaped = false, spaces = 0;
    for (const char of serialized) {
      if (escaped) {escaped = false; continue;}
      if (quoted && char === "\\") {escaped = true; continue;}
      if (char === '"') quoted = !quoted;
      if (!quoted && (char === "," || char === ":")) spaces += 1;
    }
    return encoder.encode(serialized).length + spaces;
  }
  function validateSize(body, serialized) {
    const limit = (state.model.limits || {}).max_input_bytes;
    if (limit && inputBytes(body.input) > limit) throw new Error(`输入超过此模型的 ${formatBytes(limit)} 限制，请减少文本或缩小文件。`);
    if (state.maxBodyBytes && encoder.encode(serialized).length > state.maxBodyBytes) throw new Error(`请求超过服务的 ${formatBytes(state.maxBodyBytes)} 限制，请减少输入。`);
  }
  function updateRaw() {
    $("response-details").hidden = !state.raw && !state.events.length;
    if ($("response-details").open) $("response-json").textContent = JSON.stringify(state.events.length ? {events: state.events} : state.raw, null, 2);
  }
  function metric(container, label, value) {
    const box = node("div", "stat-box");
    box.append(node("span", "stat-value", value), node("span", "stat-label", label));
    container.append(box);
  }
  function renderMedia(container, key, tag, type, extension) {
    const value = state.output[key];
    if (typeof value !== "string" || !value) return;
    try {
      const decoded = atob(value);
      const bytes = Uint8Array.from(decoded, (char) => char.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], {type}));
      state.urls.push(url);
      const media = node(tag, tag === "audio" ? "result-audio" : "result-media");
      if (tag === "img") media.alt = "模型生成的图片";
      else {media.controls = true; media.preload = "metadata";}
      media.src = url;
      const download = node("a", "download-link", `↓ 下载 ${extension.toUpperCase()}`);
      download.href = url;
      download.download = `model-output.${extension}`;
      container.append(media, download);
    } catch {container.append(node("p", "field-hint", "媒体内容无法解码，请检查完整响应 JSON。"));}
  }
  function renderResult() {
    releaseMedia();
    const container = $("result-content");
    container.replaceChildren();
    const output = state.output;
    if (typeof output.text === "string") container.append(node("div", "result-text", output.text));
    const stats = node("div", "result-stats");
    const vectors = Array.isArray(output.embeddings) ? output.embeddings : Array.isArray(output.embedding) ? [output.embedding] : null;
    if (vectors && vectors.length) {
      metric(stats, "向量数量", vectors.length);
      metric(stats, "向量维度", Array.isArray(vectors[0]) ? vectors[0].length : "—");
      container.append(stats);
      container.append(node("p", "vector-preview", `首个向量预览\n[${(vectors[0] || []).slice(0, 8).map((value) => Number(value).toFixed(5)).join(", ")}${vectors[0].length > 8 ? ", …" : ""}]`));
    } else if (Array.isArray(output.sparse_embeddings)) {
      metric(stats, "稀疏向量", output.sparse_embeddings.length);
      metric(stats, "非零项总数", output.sparse_embeddings.reduce((count, vector) => count + (Array.isArray(vector.indices) ? vector.indices.length : 0), 0));
      container.append(stats, node("p", "field-hint", "索引与权重可在完整响应 JSON 中查看。"));
    }
    if (typeof output.cosine_similarity === "number") container.append(node("p", "result-text", `声纹余弦相似度：${output.cosine_similarity.toFixed(5)}`));
    if (Array.isArray(output.results)) {
      const table = node("table", "result-table");
      const head = node("thead"); const headings = node("tr");
      ["序号", "文档", "分数"].forEach((label) => headings.append(node("th", "", label)));
      head.append(headings); table.append(head);
      const body = node("tbody");
      output.results.forEach((result) => {
        const row = node("tr");
        const documents = state.lastRequest && state.lastRequest.input.documents;
        row.append(node("td", "", result.index), node("td", "", Array.isArray(documents) ? documents[result.index] || "—" : "—"), node("td", "", Number.isFinite(result.score) ? result.score.toFixed(5) : "—"));
        body.append(row);
      });
      table.append(body); container.append(table);
    }
    renderMedia(container, "image_base64", "img", "image/png", "png");
    renderMedia(container, "audio_base64", "audio", "audio/wav", "wav");
    renderMedia(container, "video_base64", "video", "video/mp4", "mp4");
    if (!container.childNodes.length && Object.keys(output).length) container.append(node("p", "field-hint", "模型已返回结果，请展开完整响应 JSON 查看。"));
    container.hidden = !container.childNodes.length;
    if (container.childNodes.length) $("result-empty").hidden = true;
  }
  function acceptResult(data) {
    if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("响应格式无效。");
    $("result-mock").hidden = !(state.model.mock || data.mock || (data.output && data.output.mock));
    if (data.request_id) {
      $("result-meta").hidden = false;
      $("result-meta").textContent = `${((performance.now() - state.started) / 1000).toFixed(1)} s · ${data.request_id}`;
    }
    const output = data.output;
    if (output && typeof output === "object" && Object.keys(output).length) {
      const previousText = state.output.text || "";
      state.output = {...state.output, ...output};
      if (typeof output.text === "string") state.output.text = output.text;
      else if (typeof output.delta === "string") state.output.text = previousText + output.delta;
      renderResult();
    }
    updateRaw();
  }
  async function readEvents(response, signal) {
    if (!response.body) throw new Error("浏览器不支持读取流式响应。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "", completed = false;
    const processFrame = (frame) => {
      let event = "message";
      const lines = [];
      frame.split(/\r?\n/).forEach((line) => {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        if (line.startsWith("data:")) lines.push(line.slice(5).replace(/^ /, ""));
      });
      if (!lines.length) return;
      let data;
      try {data = JSON.parse(lines.join("\n"));} catch {throw new Error("服务返回的流式数据不是有效 JSON。");}
      state.events.push({event, data});
      updateRaw();
      if (event === "error" || data.error) throw apiError(data, response.status);
      acceptResult(data);
      if (data.done === true) completed = true;
    };
    try {
      while (true) {
        const {value, done} = await reader.read();
        if (signal.aborted) throw new DOMException("请求已取消", "AbortError");
        buffer += decoder.decode(value, {stream: !done});
        let match;
        while ((match = /\r?\n\r?\n/.exec(buffer))) {
          processFrame(buffer.slice(0, match.index));
          buffer = buffer.slice(match.index + match[0].length);
        }
        if (done) break;
      }
      if (buffer.trim()) processFrame(buffer);
      if (!completed) throw new Error("流式连接在完成前中断，当前内容可能不完整。请检查服务状态后再试。");
    } finally {
      // Cancelling a failed reader also closes a stream that sent an error frame.
      await reader.cancel().catch(() => {});
      reader.releaseLock();
    }
  }
  async function run(event) {
    event.preventDefault();
    if (state.busy || !state.model || state.readers.size) return;
    notice();
    let body, serialized;
    try {body = requestBody(true); serialized = JSON.stringify(body); validateSize(body, serialized);} catch (error) {notice(error.message); return;}
    resetResult();
    state.lastRequest = body;
    state.started = performance.now();
    state.busy = true;
    const generation = state.generation;
    const controller = new AbortController();
    state.controller = controller;
    refreshControls();
    resultStatus("运行中", "running");
    $("result-empty").hidden = true;
    $("result-mock").hidden = !state.model.mock;
    resultMessage("请求已发出，正在等待模型回应。首次调用可能需要加载模型。可点击「停止接收」结束等待。");
    try {
      const response = await fetch(`/v1/${state.capability}`, {method: "POST", headers: {Authorization: `Bearer ${state.key}`, "Content-Type": "application/json"}, body: serialized, credentials: "omit", cache: "no-store", signal: controller.signal});
      if (generation !== state.generation) return;
      if (!response.ok) {state.raw = await readJSON(response); updateRaw(); throw apiError(state.raw, response.status);}
      resultMessage();
      if ((response.headers.get("content-type") || "").includes("text/event-stream")) await readEvents(response, controller.signal);
      else {
        const data = await readJSON(response);
        if (generation !== state.generation) return;
        state.raw = data;
        if (data.error) {updateRaw(); throw apiError(data, response.status);}
        acceptResult(data);
      }
      if (generation !== state.generation) return;
      resultStatus("已完成", "success");
      if (!Object.keys(state.output).length) resultMessage("请求已完成，未返回可展示的内容。请查看完整响应 JSON。");
      const elapsed = ((performance.now() - state.started) / 1000).toFixed(1);
      $("result-meta").hidden = false;
      const requestId = state.raw && state.raw.request_id || state.events.find((item) => item.data.request_id)?.data.request_id;
      $("result-meta").textContent = `${elapsed} s${requestId ? ` · ${requestId}` : ""}`;
    } catch (error) {
      if (generation !== state.generation) return;
      if (error.name === "AbortError" || controller.signal.aborted) {
        resultStatus("已停止接收", "cancelled");
        resultMessage("已停止接收结果；已显示的内容可能不完整。服务仍会等待底层任务清理后再释放资源。");
      } else {
        resultStatus("运行失败", "error");
        resultMessage(error.message || "请求失败，请检查网络连接和服务状态。", "error");
      }
    } finally {
      if (generation === state.generation) {state.busy = false; state.controller = null; refreshControls(); updateRaw();}
    }
  }

  $("connection-form").addEventListener("submit", connect);
  $("api-key").addEventListener("input", () => {
    if (state.key && $("api-key").value.trim() !== state.key) {
      state.generation += 1;
      state.key = "";
      resetFiles(); clearDirectory(); resetResult(); notice();
      connectionStatus("密钥已修改，请重新连接");
    }
    refreshControls();
  });
  $("clear-key").addEventListener("click", disconnect);
  $("model-select").addEventListener("change", () => {if (!state.busy) selectModel();});
  $("form-mode").addEventListener("click", () => setMode("form"));
  $("json-mode").addEventListener("click", () => setMode("json"));
  $("json-input").addEventListener("input", updateRequestPreview);
  $("stream").addEventListener("change", updateRequestPreview);
  $("example-button").addEventListener("click", fillExample);
  $("request-details").addEventListener("toggle", updateRequestPreview);
  $("response-details").addEventListener("toggle", updateRaw);
  $("inference-form").addEventListener("submit", run);
  $("cancel-button").addEventListener("click", () => {if (state.controller) state.controller.abort();});
  window.addEventListener("pagehide", disconnect);
  refreshControls();
})();
