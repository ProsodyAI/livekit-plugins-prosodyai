# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do
not open a public issue with an API key, audio sample, transcript, or security
report.

Include the affected version, a minimal reproduction, and the impact you
observed. ProsodyAI will acknowledge the report and coordinate remediation and
disclosure privately.

## API keys and audio

Load `PROSODYAI_API_KEY` from the environment or a secret manager. The plugin
reads it in exactly one place (`GatewayConnection.from_environment`) and never
logs it; application logs and exception handlers remain the developer's
responsibility.

The key never enters the gateway URL. It is sent as the `x-api-key` header on
the WebSocket handshake, so the connection target stays safe to log, print in a
traceback, and hand to a proxy. `GatewayConnection` keeps the key out of its
`repr`.

Audio flows over one WebSocket to the configured ProsodyAI gateway for the
life of the session. Use the production endpoint unless you are testing
against a trusted local service.

The plugin exposes recording-local speaker labels and, for voices the
organization has enrolled, committed `person_id` identity facts scoped to
that organization. Raw speaker embeddings, similarity scores, and
probabilities never cross the wire.
