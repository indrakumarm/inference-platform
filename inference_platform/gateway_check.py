import http.client
import json
import signal
import ssl
import time


_CHAT = "/v1/chat/completions"


def _deadline_expired(_signal, _frame):
    raise TimeoutError("verification deadline exceeded")


def check_gateway(hostname, port, ca_file, model, token, timeout):
    context = ssl.create_default_context(cafile=str(ca_file) if ca_file else None)
    previous_handler = signal.signal(signal.SIGALRM, _deadline_expired)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        _check_auth(hostname, port, context, timeout)
        print("PASS: verified HTTPS and received HTTP 401 without a token.")
        return _check_stream(hostname, port, context, model, token, timeout)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _check_auth(hostname, port, context, timeout):
    connection = http.client.HTTPSConnection(hostname, port, context=context, timeout=timeout)
    try:
        connection.request("POST", _CHAT, body=b"{}", headers={"Content-Type": "application/json"})
        with connection.getresponse() as response:
            if response.status != 401:
                raise RuntimeError(f"Unauthenticated check: expected HTTP 401, got {response.status}")
    except ssl.SSLError as error:
        raise RuntimeError(f"HTTPS verification failed: {error}") from error
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError(f"Unauthenticated HTTPS request failed: {error}") from error
    finally:
        connection.close()


def _check_stream(hostname, port, context, model, token, timeout):
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 64, "temperature": 0, "stream": True,
    }).encode()
    connection = http.client.HTTPSConnection(hostname, port, context=context, timeout=timeout)
    start = time.monotonic()
    try:
        connection.request("POST", _CHAT, body=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {token}",
        })
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"Authenticated stream: expected HTTP 200, got {response.status}")
            if response.headers.get_content_type() != "text/event-stream":
                raise RuntimeError("Authenticated request did not return an SSE stream")
            return _read_stream(response, start)
    except TimeoutError as error:
        raise RuntimeError("Authenticated stream timed out") from error
    except (OSError, http.client.HTTPException) as error:
        raise RuntimeError(f"Authenticated stream failed ({type(error).__name__})") from error
    finally:
        connection.close()


def _read_stream(response, start):
    first_text = None
    for line in response:
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            if first_text is None:
                raise RuntimeError("Stream reached [DONE] without generated text")
            return first_text, time.monotonic() - start
        try:
            event = json.loads(data)
            for choice in event["choices"]:
                delta = choice.get("delta", {})
                if first_text is None and any(
                    isinstance(delta.get(key), str) and delta[key].strip()
                    for key in ("content", "reasoning", "reasoning_content")
                ):
                    first_text = time.monotonic() - start
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise RuntimeError("Authenticated stream contained an invalid completion event") from error
    raise RuntimeError("Authenticated stream ended without [DONE]")
