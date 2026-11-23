//! Anthropic Messages API (`/v1/messages`): a translation layer over the same machinery
//! `/v1/chat/completions` serves. Anthropic-shaped requests are rewritten to the OpenAI chat
//! shapes `prepare()` already understands, the engine runs through the same `start()`/`Gen`
//! path (sessions, drafts, LoopGuard — all of it), and answers come back in Anthropic's
//! response and SSE event shapes.
//!
//! Translated: system (string or text blocks), text blocks, assistant `tool_use` blocks (→
//! OpenAI `tool_calls`), user `tool_result` blocks (→ `role:"tool"` messages), `tools`
//! (`input_schema` → `parameters`), `tool_choice`, `thinking` (budget → reasoning effort),
//! `stop_sequences`, temperature/top_p/top_k, max_tokens, stream.
//!
//! Images translate too (Anthropic `image` blocks -> image_url parts; base64 sources
//! become data URLs). Not yet: `/v1/messages/count_tokens`, the legacy `/v1/complete`.

use crate::encoding::{Delta, ThinkingMode};
use crate::gate::{Refused, RETRY_AFTER_S};
use crate::server::{App, check_auth, prepare, start, Finish, Prepared, Usage, KEEP_ALIVE};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Map, Value};
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------- handler

pub(crate) async fn messages(State(app): State<Arc<App>>, headers: HeaderMap, body: Bytes) -> Response {
    if let Err(r) = check_auth(&app, &headers) {
        // the chat path's 401 body; an Anthropic-shaped one is the same status and idea
        return aerror(StatusCode::UNAUTHORIZED, "authentication_error", "invalid API key");
    }
    let mut req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return aerror(StatusCode::BAD_REQUEST, "invalid_request_error", format!("invalid JSON: {e}")),
    };
    drop(body);
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&app.cfg.model_name)
        .to_string();
    if req.get("max_tokens").and_then(Value::as_u64).is_none() {
        return aerror(StatusCode::BAD_REQUEST, "invalid_request_error", "max_tokens is required");
    }
    // thinking: Anthropic's {type:"enabled", budget_tokens:N} is already a thinking switch
    // prepare() understands ({type: enabled}); the budget is a different unit than this
    // server's 1-100 effort, so it becomes a named effort tier. {type:"disabled"} switches
    // thinking off and stays as-is (an explicit off-switch wins in prepare()).
    let thinking_budget = req.get("thinking").and_then(|t| {
        if t.get("type").and_then(Value::as_str) == Some("enabled") {
            t.get("budget_tokens").and_then(Value::as_u64)
        } else {
            None
        }
    });
    let openai = match translate(&req, thinking_budget) {
        Ok(v) => v,
        Err(e) => return aerror(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let mut p = match prepare(&app.cfg, &openai) {
        Ok(p) => p,
        Err(e) => return aerror(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let wants_thinking = p.mode == ThinkingMode::Thinking;
    // A stream that has to wait starts at once with SSE comments (a client's idle timeout sees
    // bytes while the engine is away); refused, it ends with an error event. Anything else
    // waits here and is refused with 503 + Retry-After.
    if p.stream && !app.gate.open(app.engine.info().is_some()) {
        return stream_messages(app, p, model, wants_thinking, None);
    }
    let pass = match app.gate.enter(&app.engine).await {
        Ok(pass) => pass,
        Err(r) => {
            let mut resp = refused_response(r);
            resp.headers_mut().insert("retry-after", RETRY_AFTER_S.into());
            return resp;
        }
    };
    if p.stream {
        return stream_messages(app, p, model, wants_thinking, Some(pass));
    }

    let mut gen = match start(&app, &p, pass).await {
        Ok(g) => g,
        Err(e) => {
            let (status, kind, msg) = api_parts(e);
            return aerror(status, anthropic_kind(&kind), msg);
        }
    };
    let id = format!("msg_{}", app.random_hex(24));
    let mut out = Vec::new();
    let (mut reasoning, mut content, mut calls) = (String::new(), String::new(), Vec::new());
    let (finish, usage) = loop {
        let end = gen.next(&mut out).await;
        for d in out.drain(..) {
            match d {
                Delta::Reasoning(s) => reasoning.push_str(&s),
                Delta::Content(s) => content.push_str(&s),
                Delta::ToolCall(_, name, args) => calls.push((name, args)),
            }
        }
        match end {
            None => continue,
            Some(Ok(x)) => break x,
            Some(Err(e)) => return aerror(StatusCode::INTERNAL_SERVER_ERROR, "api_error", e),
        }
    };
    let mut content_blocks = Vec::new();
    if wants_thinking && !reasoning.is_empty() {
        content_blocks.push(json!({"type": "thinking", "thinking": reasoning}));
    }
    if !content.is_empty() || calls.is_empty() {
        content_blocks.push(json!({"type": "text", "text": content}));
    }
    for (name, args) in &calls {
        content_blocks.push(tool_use_block(&app, name, args));
    }
    let body = json!({
        "id": id, "type": "message", "role": "assistant", "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason(finish), "stop_sequence": Value::Null,
        "usage": anthropic_usage(&usage),
    });
    (StatusCode::OK, [("content-type", "application/json")], body.to_string()).into_response()
}

// ---------------------------------------------------------------- streaming

/// One streamed content block. Blocks are sequential: start, deltas, stop. A tool_use block
/// is born complete (one tool-call delta carries the whole arguments string).
struct OpenBlock {
    index: usize,
    kind: BlockKind,
    id: String,
    name: String,
}

#[derive(Clone, Copy, PartialEq)]
enum BlockKind {
    Text,
    Thinking,
    ToolUse,
}

impl BlockKind {
    fn start_type(self) -> &'static str {
        match self {
            BlockKind::Text => "text",
            BlockKind::Thinking => "thinking",
            BlockKind::ToolUse => "tool_use",
        }
    }
    fn delta_type(self) -> &'static str {
        match self {
            BlockKind::Text => "text_delta",
            BlockKind::Thinking => "thinking_delta",
            BlockKind::ToolUse => "input_json_delta",
        }
    }
}

/// A streamed answer. Without a pass it waits for the engine first: the stream starts at once
/// and carries comments meanwhile, and a refusal is its error event (the chat path's shape).
fn stream_messages(
    app: Arc<App>,
    p: Prepared,
    model: String,
    wants_thinking: bool,
    pass: Option<crate::gate::Pass>,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let aerror_event = |tx: tokio::sync::mpsc::Sender<Event>, status: StatusCode, kind: String, msg: String| async move {
            let _ = status;
            let _ = tx
                .send(Event::default().data(json!({"type": "error", "error": {"type": kind, "message": msg}}).to_string()))
                .await;
            let _ = tx.send(Event::default().data("[DONE]")).await;
        };
        let pass = if let Some(pass) = pass {
            pass
        } else {
            tokio::select! {
                r = app.gate.enter(&app.engine) => match r {
                    Ok(pass) => pass,
                    Err(r) => {
                        let (t, msg) = refused_parts(r);
                        return aerror_event(tx, t, "api_error".to_string(), msg).await;
                    }
                },
                _ = tx.closed() => return, // the client went away: counted out at the gate
            }
        };
        let mut gen = match start(&app, &p, pass).await {
            Ok(g) => g,
            Err(e) => {
                let (status, kind, msg) = api_parts(e);
                return aerror_event(tx, status, kind, msg).await;
            }
        };
        let id = format!("msg_{}", app.random_hex(24));
        let mut send = |v: Value| {
            let tx = tx.clone();
            async move { tx.send(Event::default().data(v.to_string())).await.is_err() }
        };
        if send(json!({"type": "message_start", "message": {"id": id, "type": "message", "role": "assistant",
                           "model": model, "content": [], "stop_reason": Value::Null, "stop_sequence": Value::Null,
                           "usage": {"input_tokens": 0, "output_tokens": 0}}}))
            .await
        {
            return; // client gone
        }
        let mut out = Vec::new();
        let mut block: Option<OpenBlock> = None; // at most one open block at a time
        let mut next_index = 0usize;
        let mut input_tokens = 0usize;
        let mut output_tokens = 0usize;
        loop {
            let end = gen.next(&mut out).await;
            let mut client_gone = false;
            for d in out.drain(..) {
                let kind: BlockKind = match d {
                    Delta::Reasoning(ref s) => {
                        if !wants_thinking {
                            continue;
                        }
                        BlockKind::Thinking
                    }
                    Delta::Content(ref s) => BlockKind::Text,
                    Delta::ToolCall(ref _i, ref name, ref args) => BlockKind::ToolUse,
                };
                // close a block of a different kind, then open the new one
                let same = block.as_ref().map(|b| b.kind) == Some(kind);
                if !same {
                    if let Some(b) = block.take() {
                        client_gone |= send(json!({"type": "content_block_stop", "index": b.index})).await;
                    }
                }
                if block.is_none() {
                    let idx = next_index;
                    next_index += 1;
                    let (bid, bname) = match (&d, kind) {
                        (Delta::ToolCall(_, name, _), BlockKind::ToolUse) => {
                            (format!("toolu_{}", app.random_hex(24)), name.clone())
                        }
                        _ => (String::new(), String::new()),
                    };
                    let content_block = match kind {
                        BlockKind::ToolUse => json!({"type": "tool_use", "id": bid, "name": bname, "input": {}}),
                        BlockKind::Thinking => json!({"type": "thinking", "thinking": ""}),
                        BlockKind::Text => json!({"type": "text", "text": ""}),
                    };
                    block = Some(OpenBlock { index: idx, kind, id: bid, name: bname });
                    client_gone |= send(json!({"type": "content_block_start", "index": idx, "content_block": content_block})).await;
                }
                let delta_payload = match (&d, kind) {
                    (Delta::Reasoning(s), _) => json!({"type": kind.delta_type(), "thinking": s}),
                    (Delta::Content(s), _) => json!({"type": kind.delta_type(), "text": s}),
                    (Delta::ToolCall(_, _, args), _) => json!({"type": kind.delta_type(), "partial_json": args}),
                };
                let idx = block.as_ref().map(|b| b.index).unwrap_or(0);
                client_gone |= send(json!({"type": "content_block_delta", "index": idx, "delta": delta_payload})).await;
                if client_gone {
                    return; // client gone; dropping gen cancels the request
                }
            }
            match end {
                None => continue,
                Some(Ok((finish, usage))) => {
                    input_tokens = usage.prompt;
                    output_tokens = usage.completion;
                    if let Some(b) = block.take() {
                        if send(json!({"type": "content_block_stop", "index": b.index})).await {
                            return;
                        }
                    }
                    if send(json!({"type": "message_delta", "delta": {"stop_reason": stop_reason(finish), "stop_sequence": Value::Null},
                                   "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}))
                        .await
                    {
                        return;
                    }
                    let _ = send(json!({"type": "message_stop"})).await;
                    break;
                }
                Some(Err(e)) => {
                    let _ = send(json!({"type": "error", "error": {"type": "api_error", "message": e}})).await;
                    break;
                }
            }
        }
    });
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let stream = futures_util::StreamExt::map(stream, Ok::<_, Infallible>);
    Sse::new(stream).keep_alive(KeepAlive::new().interval(KEEP_ALIVE)).into_response()
}

// ---------------------------------------------------------------- translation

/// Anthropic request → the OpenAI chat request `prepare()` serves. Consumes `thinking`
/// (adds the effort tier next to it, keeping the object as the switch) and rewrites the
/// content blocks; everything else (temperature, top_p, top_k, stream, metadata…) rides along.
fn translate(req: &Value, thinking_budget: Option<u64>) -> Result<Value, String> {
    let o = req.as_object().ok_or("the request body must be a JSON object")?;
    let mut out = Map::new();
    for (k, v) in o {
        if k == "system" || k == "messages" || k == "tools" || k == "tool_choice" || k == "thinking" {
            continue; // rewritten below
        }
        out.insert(k.clone(), v.clone());
    }
    if let Some(ss) = o.get("stop_sequences").cloned() {
        out.insert("stop".into(), ss);
    }
    if thinking_budget.is_some() {
        let effort = match thinking_budget.unwrap() {
            b if b < 4_096 => "low",
            b if b < 16_384 => "high",
            _ => "max",
        };
        out.insert("reasoning_effort".into(), json!(effort));
        // the object itself is a valid thinking switch for prepare() ({type: enabled});
        // keep it so the request still switches thinking on
        out.insert("thinking".into(), req.get("thinking").cloned().unwrap_or(json!({"type": "enabled"})));
    }
    if let Some(s) = o.get("system") {
        out.insert("messages".into(), Value::Array(Vec::new())); // placeholder, filled below
        let sys_text = match s {
            Value::String(t) => t.clone(),
            Value::Array(blocks) => blocks_text(blocks)?,
            other => return Err(format!("unsupported system value: {other}")),
        };
        let msgs = out.get_mut("messages").and_then(Value::as_array_mut).unwrap();
        msgs.push(json!({"role": "system", "content": sys_text}));
        if let Some(ms) = o.get("messages").and_then(Value::as_array) {
            for m in ms {
                translate_message(m, msgs)?;
            }
        }
    } else {
        let mut msgs = Vec::new();
        if let Some(ms) = o.get("messages").and_then(Value::as_array) {
            for m in ms {
                translate_message(m, &mut msgs)?;
            }
        }
        out.insert("messages".into(), Value::Array(msgs));
    }
    if let Some(Value::Array(ts)) = o.get("tools") {
        let mut fun = Vec::new();
        for t in ts {
            let name = t.get("name").and_then(Value::as_str).ok_or("a tool entry has no name")?;
            fun.push(json!({"type": "function", "function": {
                "name": name,
                "description": t.get("description").cloned().unwrap_or(Value::Null),
                "parameters": t.get("input_schema").cloned().unwrap_or(json!({"type": "object"})),
            }}));
        }
        out.insert("tools".into(), Value::Array(fun));
    }
    if let Some(tc) = o.get("tool_choice") {
        let choice = match tc.get("type").and_then(Value::as_str) {
            Some("auto") => json!("auto"),
            Some("any") => json!("required"),
            Some("tool") => {
                let name = tc.get("name").and_then(Value::as_str).ok_or("tool_choice of type \"tool\" has no name")?;
                json!({"type": "function", "function": {"name": name}})
            }
            Some("none") => json!("none"),
            other => return Err(format!("unsupported tool_choice type: {other:?}")),
        };
        out.insert("tool_choice".into(), choice);
    }
    Ok(Value::Object(out))
}

/// One Anthropic message → one or more OpenAI messages appended to `out`.
fn translate_message(m: &Value, out: &mut Vec<Value>) -> Result<(), String> {
    let role = m.get("role").and_then(Value::as_str).ok_or("a message has no role")?;
    let content = m.get("content");
    match (role, content) {
        ("assistant", Some(Value::Array(blocks))) => {
            let mut texts = Vec::new();
            let mut calls = Vec::new();
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => texts.push(b.get("text").and_then(Value::as_str).unwrap_or("")),
                    Some("thinking") => {} // history reasoning: the encoder drops it
                    Some("tool_use") => {
                        let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                        let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                        let input = b.get("input").cloned().unwrap_or(json!({}));
                        calls.push(json!({"id": id, "type": "function", "function": {
                            "name": name, "arguments": Value::String(input.to_string())}}));
                    }
                    Some(other) => return Err(format!("unsupported assistant content block: {other}")),
                    None => return Err("an assistant content block has no type".into()),
                }
            }
            let mut msg = json!({"role": "assistant", "content": texts.join("")});
            if !calls.is_empty() {
                msg["tool_calls"] = Value::Array(calls);
                if texts.is_empty() {
                    msg["content"] = Value::Null;
                }
            }
            out.push(msg);
        }
        (role, Some(Value::Array(blocks))) if role == "user" => {
            // tool_result blocks become role:"tool" messages; text rides in the user message.
            // Order is kept: a tool_result belongs after the assistant turn that called it.
            // Image blocks become image_url parts of the user message (the vision path).
            let mut texts: Vec<String> = Vec::new();
            let mut images: Vec<Value> = Vec::new();
            let flush = |texts: &mut Vec<String>, images: &mut Vec<Value>, out: &mut Vec<Value>| {
                if !images.is_empty() {
                    let mut parts = Vec::new();
                    if !texts.is_empty() {
                        parts.push(json!({"type": "text", "text": texts.join("")}));
                    }
                    parts.extend(images.drain(..));
                    out.push(json!({"role": "user", "content": parts}));
                } else if !texts.is_empty() {
                    out.push(json!({"role": "user", "content": texts.join("")}));
                    texts.clear();
                }
            };
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(t) = b.get("text").and_then(Value::as_str) {
                            texts.push(t.to_string());
                        }
                    }
                    Some("image") | Some("image_url") => {
                        images.push(image_url_part(b)?);
                    }
                    Some("tool_result") => {
                        flush(&mut texts, &mut images, out);
                        let call_id = b.get("tool_use_id").and_then(Value::as_str).unwrap_or("");
                        let content = match b.get("content") {
                            None => "".to_string(),
                            Some(Value::String(t)) => t.clone(),
                            Some(Value::Array(parts)) => {
                                let mut parts_s = Vec::new();
                                for p in parts {
                                    match p.get("type").and_then(Value::as_str) {
                                        Some("text") => parts_s.push(p.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                                        other => parts_s.push(format!("[Unsupported {:?}]", other)),
                                    }
                                }
                                parts_s.join("\n")
                            }
                            Some(other) => other.to_string(),
                        };
                        let is_err = truthy_b(b.get("is_error"));
                        let mut msg = json!({"role": "tool", "tool_call_id": call_id, "content": content});
                        if is_err {
                            msg["is_error"] = json!(true);
                        }
                        out.push(msg);
                    }
                    Some(other) => return Err(format!("unsupported user content block: {other}")),
                    None => return Err("a user content block has no type".into()),
                }
            }
            flush(&mut texts, &mut images, out);
        }
        (role, Some(Value::Array(_))) => return Err(format!("unsupported message role with content blocks: {role}")),
        (role, Some(Value::String(t))) => out.push(json!({"role": role, "content": t})),
        (role, None) => out.push(json!({"role": role, "content": ""})),
        (_, other) => return Err(format!("unsupported message content: {other:?}")),
    }
    Ok(())
}

/// Text blocks of an array (a system prompt's blocks): strings joined, anything else an error.
fn blocks_text(blocks: &[Value]) -> Result<String, String> {
    let arr = blocks;
    let mut out = Vec::new();
    for b in arr {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => out.push(b.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
            other => return Err(format!("unsupported system content block: {other:?}")),
        }
    }
    Ok(out.join("\n"))
}


/// An Anthropic image block -> the image_url part the vision path serves
/// ({type: base64, media_type, data} -> a data URL; {type: url, url} -> the URL).
fn image_url_part(b: &Value) -> Result<Value, String> {
    if let Some(iu) = b.get("image_url") {
        return Ok(json!({"type": "image_url", "image_url": iu.clone()}));
    }
    let src = b.get("source").ok_or("an image block has no source")?;
    match src.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let mt = src.get("media_type").and_then(Value::as_str).unwrap_or("image/png");
            let data = src.get("data").and_then(Value::as_str).ok_or("a base64 image source has no data")?;
            Ok(json!({"type": "image_url", "image_url": {"url": format!("data:{mt};base64,{data}")}}))
        }
        Some("url") => {
            let url = src.get("url").and_then(Value::as_str).ok_or("a url image source has no url")?;
            Ok(json!({"type": "image_url", "image_url": {"url": url}}))
        }
        other => Err(format!("unsupported image source type: {other:?}")),
    }
}

fn truthy_b(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Bool(true)))
}

fn parse_input(args: &str) -> Value {
    serde_json::from_str(args).unwrap_or(Value::String(args.to_string()))
}

fn stop_reason(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "end_turn",
        Finish::Length => "max_tokens",
        Finish::ToolCalls => "tool_use",
    }
}

fn tool_use_block(app: &App, name: &str, args: &str) -> Value {
    json!({"type": "tool_use", "id": format!("toolu_{}", app.random_hex(24)), "name": name, "input": parse_input(args)})
}

fn anthropic_usage(u: &Usage) -> Value {
    let mut v = json!({"input_tokens": u.prompt, "output_tokens": u.completion});
    if u.cached > 0 {
        v["cache_read_input_tokens"] = json!(u.cached);
    }
    v
}

// ---------------------------------------------------------------- errors and refusals

fn anthropic_kind(kind: &str) -> &'static str {
    match kind {
        "authentication_error" => "authentication_error",
        "invalid_request_error" | "rate_limit_error" | "permission_error" | "not_found_error" => "invalid_request_error",
        _ => "api_error",
    }
}

fn aerror<S: Into<String>, M: Into<String>>(status: StatusCode, kind: S, msg: M) -> Response {
    let body = json!({"type": "error", "error": {"type": kind.into(), "message": msg.into()}});
    (status, [("content-type", "application/json")], body.to_string()).into_response()
}

/// (status, Anthropic error type, message) out of the chat path's ApiError.
fn api_parts(e: crate::server::ApiError) -> (StatusCode, String, String) {
    let (status, kind, msg) = e.parts();
    let k = anthropic_kind(&kind).to_string();
    (status, k, msg)
}

fn refused_parts(r: Refused) -> (StatusCode, String) {
    match r {
        Refused::Held(t) => (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("the engine has been down for the {} s this request waited (the ring is lent to a development job, or the engine restarts); try again later", t.as_secs()),
        ),
        Refused::Closing => (StatusCode::SERVICE_UNAVAILABLE, "the server is shutting down".into()),
    }
}

fn refused_response(r: Refused) -> Response {
    let (status, msg) = refused_parts(r);
    aerror(status, "api_error", msg)
}


// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn system_string_and_blocks_become_a_system_message() {
        let req = json!({"model": "m", "max_tokens": 16, "system": "be terse",
                         "messages": [{"role": "user", "content": "hi"}]});
        let out = translate(&req, None).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "be terse");
        assert_eq!(msgs[1]["content"], "hi");

        let req = json!({"model": "m", "max_tokens": 16, "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
                         "messages": [{"role": "user", "content": "hi"}]});
        let out = translate(&req, None).unwrap();
        assert_eq!(out["messages"][0]["content"], "a\nb");
    }

    #[test]
    fn tool_use_and_tool_result_become_openai_tools() {
        let req = json!({"model": "m", "max_tokens": 16,
            "tools": [{"name": "get_time", "description": "clock", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "what time?"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "12:00"}]},
            ]});
        let out = translate(&req, None).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_time");
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["arguments"], "{}");
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "toolu_1");
        assert_eq!(msgs[2]["content"], "12:00");
        assert_eq!(out["tools"][0]["type"], "function");
        assert_eq!(out["tools"][0]["function"]["parameters"], json!({"type": "object"}));
    }

    #[test]
    fn thinking_enabled_becomes_an_effort_and_stays_a_switch() {
        let req = json!({"model": "m", "max_tokens": 16, "thinking": {"type": "enabled", "budget_tokens": 2048},
                         "messages": [{"role": "user", "content": "hi"}]});
        let out = translate(&req, Some(2048)).unwrap();
        assert_eq!(out["reasoning_effort"], "low");
        assert_eq!(out["thinking"]["type"], "enabled");
    }

    #[test]
    fn tool_choice_translates() {
        let base = json!({"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "x"}]});
        let mut req = base.clone();
        req["tool_choice"] = json!({"type": "auto"});
        assert_eq!(translate(&req, None).unwrap()["tool_choice"], "auto");
        let mut req = base.clone();
        req["tool_choice"] = json!({"type": "any"});
        assert_eq!(translate(&req, None).unwrap()["tool_choice"], "required");
        let mut req = base.clone();
        req["tool_choice"] = json!({"type": "tool", "name": "pick"});
        assert_eq!(translate(&req, None).unwrap()["tool_choice"]["function"]["name"], "pick");
    }

    #[test]
    fn stop_sequences_and_scalars_pass_through() {
        let req = json!({"model": "m", "max_tokens": 64, "temperature": 0.5, "stop_sequences": ["END"],
                         "stream": true, "messages": [{"role": "user", "content": "x"}]});
        let out = translate(&req, None).unwrap();
        assert_eq!(out["stop"], json!(["END"]));
        assert_eq!(out["temperature"], 0.5);
        assert_eq!(out["stream"], true);
    }

    #[test]
    fn image_block_becomes_an_image_url_part() {
        let req = json!({"model": "m", "max_tokens": 16, "system": "describe",
                         "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "what is this?"},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aa"}}]}]});
        let out = translate(&req, None).unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        let content = msgs[1]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,aa");
    }

    #[test]
    fn stop_reason_maps() {
        assert_eq!(stop_reason(Finish::Stop), "end_turn");
        assert_eq!(stop_reason(Finish::Length), "max_tokens");
        assert_eq!(stop_reason(Finish::ToolCalls), "tool_use");
    }
}
