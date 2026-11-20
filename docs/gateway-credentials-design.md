# Gateway credentials: token management and certificate provisioning

## Decision

Automate bearer-token installation and rotation independently of certificate
issuance and renewal. Defer certificate automation, not the requirement for
encrypted transport. A VPN restricts network access, but does not replace
HTTPS with server identity verification at the gateway. A bearer token is a
replayable credential: anyone who obtains it can use it until it is revoked.

The gateway enforces bearer authentication in its HTTPS mode. VPN clients
must trust the CA-issued certificate or explicitly trust the lab certificate.
For operator-only work without a certificate, keep the listener on
loopback and use SSH forwarding. That mode does not currently enforce bearer
authentication. Do not add a bearer-only HTTP listener on the VPN interface.

## Implemented: one-time lab bootstrap

`gateway bootstrap-tls` takes the hostname from an HTTPS gateway profile,
prompts for a bearer token without echoing it, and generates a 30-day
self-signed certificate with OpenSSL. It installs new credentials only and
leaves NGINX activation to `gateway deploy`. Clients explicitly trust the
public lab certificate obtained over trusted SSH. Certbot is not needed.

### HTTPS deployment observations

The operator reported completing the migration to standard HTTPS port `443`.
The client URL no longer needs a port, and vLLM still listens on loopback
port `8000`. Changing the listener port did not require a new certificate
or token.

On the earlier HTTPS listener at port `8080`, curl verified the lab
certificate and received `401` without a bearer token. A local NGINX test
with a mock backend also verified authenticated streaming and that the token
was not forwarded upstream. If not already done on the live `443` listener,
verify both a `401` without a token and a stream ending in `[DONE]` with it.
See the [HTTPS setup and smoke test](gateway.md#https-and-bearer-authentication-for-vpn-clients).

## Future implementation: independent token management

A proposed `gateway token install` command would read an operator-provided
token from stdin or a protected file, validate its format, and write the
root-only NGINX token map without printing the secret. Keep tokens out of
profiles, command-line arguments, logs, and the repository.

- **First setup:** install the token map before applying the HTTPS gateway
  profile. The certificate and key can continue to be provisioned manually.
- **Rotation:** replace the map atomically, validate the active NGINX
  configuration, and reload only on success. Restore the previous map if
  validation or the reload command fails. Verify that new requests accept the
  replacement token and reject the retired one.

Token management does not create certificates, change listeners, or modify
the upstream pool. Keep it separate from `gateway deploy`.

## Parked: certificate provisioning and renewal

Choose a certificate source when this work resumes:

| Mode | Issuance and renewal |
|---|---|
| Import | Install a certificate and key from a trusted issuer. The operator or issuer supplies the renewed pair. |
| OpenSSL CSR | Generate the private key and CSR on the gateway. An internal CA signs the CSR; install the returned chain. OpenSSL does not automate CA issuance or renewal. |
| Certbot / ACME | Use `certonly` against the selected ACME CA. A successful renewal runs a deploy hook; Certbot does not edit the platform-managed NGINX site. |

All three modes should share one activation path: validate the hostname,
trust chain, expiry, and key match; stage the pair together; validate and
reload NGINX; then verify the certificate actually served. Retain the previous
pair for rollback. The gateway profile remains the source of truth for the
hostname. Add `cert.json` only if there is persistent issuer or challenge
policy to capture; never store keys or challenge credentials in it.
