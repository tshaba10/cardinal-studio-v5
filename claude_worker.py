"""Isolated streaming HTTPS worker. No response deadline; parent can terminate it."""

import json
import os
import sys
import time
import requests


def emit(value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def stream_response(body, key, notify=emit):
    body = dict(body, stream=True)
    # Only connection establishment is bounded. Stop kills even a blocked socket read.
    with requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        stream=True,
        timeout=(10, None),
    ) as response:
        if not response.ok:
            hints = {
                400: "Check the model ID and request structure.",
                401: "Check ANTHROPIC_API_KEY in .env.",
                403: "Check organization and model permissions.",
                404: "Check the model ID.",
                429: "Check API credits and rate limits.",
                529: "Claude is busy. Retry deliberately later.",
            }
            raise ValueError(
                f"Claude HTTP {response.status_code}. "
                + hints.get(response.status_code, "Check service availability.")
            )
        notify({"progress": "Connected to Claude; waiting for response events"})
        parts, usage, reason, started, finished = [], {}, None, False, False
        characters, last_note, data_lines = 0, 0.0, []
        response.encoding = "utf-8"

        def event(data):
            nonlocal reason, started, finished, characters, last_note
            kind = data.get("type")
            if kind == "error":
                raise ValueError(
                    "Claude reported a streaming error. No partial answer was applied. Retry deliberately; no automatic retry was made."
                )
            if kind == "message_start":
                started = True
                usage.update(data.get("message", {}).get("usage", {}))
            elif kind == "content_block_start":
                block = data.get("content_block", {})
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                    characters += len(parts[-1])
            elif kind == "content_block_delta":
                delta = data.get("delta", {})
                if delta.get("type") == "text_delta":
                    parts.append(delta.get("text", ""))
                    characters += len(parts[-1])
            elif kind == "message_delta":
                reason = data.get("delta", {}).get("stop_reason", reason)
                usage.update(data.get("usage", {}))
            elif kind == "message_stop":
                finished = True
            if characters > 2_000_000:
                raise ValueError(
                    "The response exceeded the safe output-size limit. No partial answer was accepted. Split the design into smaller scopes."
                )
            if time.monotonic() - last_note >= 0.5:
                notify(
                    {
                        "progress": f"Receiving Claude response · {characters:,} text characters received"
                        if characters
                        else "Claude connection active; waiting for response text"
                    }
                )
                last_note = time.monotonic()

        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if line == "":
                if data_lines:
                    event(json.loads("\n".join(data_lines)))
                    data_lines = []
                if finished:
                    break
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines and not finished:
            event(json.loads("\n".join(data_lines)))
        if not started or not finished or reason != "end_turn":
            raise ValueError(
                "Claude returned an incomplete or refused answer. No partial design was applied. Review the request or output size before retrying."
            )
        return dict(text="".join(parts), usage=usage)


def main():
    try:
        emit({"result": stream_response(json.load(sys.stdin), os.environ["ANTHROPIC_API_KEY"])})
    except requests.exceptions.SSLError:
        emit(
            {
                "error": "Claude certificate verification failed. Use company-approved certificate settings; never disable TLS checks."
            }
        )
    except requests.exceptions.ConnectTimeout:
        emit(
            {
                "error": "Could not establish a Claude connection within 10 seconds. Check your approved network/proxy; no automatic retry was made."
            }
        )
    except requests.exceptions.RequestException:
        emit(
            {
                "error": "The Claude network connection failed or was interrupted. No partial answer was applied. Check your approved network/proxy."
            }
        )
    except ValueError as exc:
        emit(
            {
                "error": str(exc)
                if str(exc).startswith(("Claude ", "The response "))
                else "Claude returned an unreadable stream. No partial answer was applied."
            }
        )
    except Exception:
        emit(
            {
                "error": "The Claude worker could not complete the request. Check local setup. No partial answer was applied."
            }
        )


if __name__ == "__main__":
    main()
