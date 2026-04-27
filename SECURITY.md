# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Do not open public issues for security vulnerabilities.**

Instead, please email **robert.dominic.vitali@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Your suggested fix (if any)

### Response Timeline

- **Initial response**: within 72 hours
- **Assessment and fix target**: within 30 days
- **Public disclosure**: coordinated after the fix is released

We appreciate responsible disclosure and will credit reporters in the release
notes (unless you prefer to remain anonymous).

## Security Considerations

ollama-marshal is designed to run on a local machine as a proxy between
local programs and a local Ollama instance. It does not implement
authentication or encryption by default, as it is intended for
single-machine use behind a firewall.

If you expose ollama-marshal to a network:

- Use a reverse proxy (nginx, Caddy) with TLS termination
- Add authentication at the reverse proxy layer
- Restrict access by IP or network segment
