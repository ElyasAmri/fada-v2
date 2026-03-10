#!/bin/bash
# Create a Cloudflare Tunnel for fada-monitor and deploy to remote.
# Reads credentials from .env.local, creates tunnel via CF API,
# then deploys tunnel token to fada-1 via Ansible.
#
# Usage: bash experiments/rccg/fada-monitor/setup_tunnel.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ENV_FILE="$PROJECT_DIR/.env.local"

TUNNEL_NAME="fada-monitor"
HOSTNAME="fada-monitor.elyasamri.com"
LOCAL_SERVICE="http://localhost:9731"

# Load credentials from .env.local
if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found"
  exit 1
fi

CF_API_TOKEN=$(grep '^CLOUDFLARE_API_TOKEN=' "$ENV_FILE" | cut -d= -f2- | tr -d '\r\n')
CF_ACCOUNT_ID=$(grep '^CLOUDFLARE_ACCOUNT_ID=' "$ENV_FILE" | cut -d= -f2- | tr -d '\r\n')
CF_ZONE_ID=$(grep '^CLOUDFLARE_ZONE_ID=' "$ENV_FILE" | cut -d= -f2- | tr -d '\r\n')

if [ -z "$CF_API_TOKEN" ] || [ -z "$CF_ACCOUNT_ID" ]; then
  echo "ERROR: Missing CLOUDFLARE_API_TOKEN or CLOUDFLARE_ACCOUNT_ID in .env.local"
  exit 1
fi

echo "=== Setting up Cloudflare Tunnel ==="
echo "Tunnel: $TUNNEL_NAME"
echo "Hostname: $HOSTNAME"
echo "Service: $LOCAL_SERVICE"
echo

# Step 1: Check if tunnel already exists
echo "[1/4] Checking for existing tunnel..."
EXISTING=$(curl -sf \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/cfd_tunnel?name=$TUNNEL_NAME&is_deleted=false" \
  2>/dev/null)

TUNNEL_ID=$(echo "$EXISTING" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -n "$TUNNEL_ID" ]; then
  echo "  Found existing tunnel: $TUNNEL_ID"
else
  # Step 2: Create tunnel
  echo "[2/4] Creating tunnel..."
  CREATE_RESP=$(curl -sf \
    -X POST \
    -H "Authorization: Bearer $CF_API_TOKEN" \
    -H "Content-Type: application/json" \
    "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/cfd_tunnel" \
    -d "{\"name\":\"$TUNNEL_NAME\",\"tunnel_secret\":\"$(openssl rand -base64 32)\"}")

  TUNNEL_ID=$(echo "$CREATE_RESP" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

  if [ -z "$TUNNEL_ID" ]; then
    echo "ERROR: Failed to create tunnel"
    echo "$CREATE_RESP"
    exit 1
  fi
  echo "  Created tunnel: $TUNNEL_ID"
fi

# Step 3: Configure tunnel routing
echo "[3/4] Configuring tunnel route..."
CONFIG_RESP=$(curl -sf \
  -X PUT \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  -H "Content-Type: application/json" \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/configurations" \
  -d "{\"config\":{\"ingress\":[{\"hostname\":\"$HOSTNAME\",\"service\":\"$LOCAL_SERVICE\"},{\"service\":\"http_status:404\"}]}}")

echo "  Route configured: $HOSTNAME -> $LOCAL_SERVICE"

# Step 4: Create DNS record (CNAME to tunnel)
echo "[4/4] Creating DNS record..."
if [ -n "$CF_ZONE_ID" ]; then
  # Check if DNS record exists
  DNS_CHECK=$(curl -sf \
    -H "Authorization: Bearer $CF_API_TOKEN" \
    "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records?name=$HOSTNAME&type=CNAME")

  DNS_ID=$(echo "$DNS_CHECK" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

  if [ -n "$DNS_ID" ]; then
    # Update existing
    curl -sf \
      -X PUT \
      -H "Authorization: Bearer $CF_API_TOKEN" \
      -H "Content-Type: application/json" \
      "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records/$DNS_ID" \
      -d "{\"type\":\"CNAME\",\"name\":\"fada-monitor\",\"content\":\"$TUNNEL_ID.cfargotunnel.com\",\"proxied\":true}" >/dev/null
    echo "  Updated DNS: $HOSTNAME -> $TUNNEL_ID.cfargotunnel.com"
  else
    # Create new
    curl -sf \
      -X POST \
      -H "Authorization: Bearer $CF_API_TOKEN" \
      -H "Content-Type: application/json" \
      "https://api.cloudflare.com/client/v4/zones/$CF_ZONE_ID/dns_records" \
      -d "{\"type\":\"CNAME\",\"name\":\"fada-monitor\",\"content\":\"$TUNNEL_ID.cfargotunnel.com\",\"proxied\":true}" >/dev/null
    echo "  Created DNS: $HOSTNAME -> $TUNNEL_ID.cfargotunnel.com"
  fi
else
  echo "  SKIP: No CLOUDFLARE_ZONE_ID, create CNAME manually:"
  echo "    fada-monitor CNAME $TUNNEL_ID.cfargotunnel.com"
fi

# Get tunnel token
echo
echo "=== Getting tunnel token ==="
TOKEN_RESP=$(curl -sf \
  -H "Authorization: Bearer $CF_API_TOKEN" \
  "https://api.cloudflare.com/client/v4/accounts/$CF_ACCOUNT_ID/cfd_tunnel/$TUNNEL_ID/token")

TUNNEL_TOKEN=$(echo "$TOKEN_RESP" | grep -o '"result":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TUNNEL_TOKEN" ]; then
  echo "ERROR: Could not get tunnel token"
  echo "$TOKEN_RESP" | sed 's/"result":"[^"]*"/"result":"[REDACTED]"/g'
  exit 1
fi

echo "Token retrieved (${#TUNNEL_TOKEN} chars)"

# Save token for Ansible
TOKEN_FILE="$SCRIPT_DIR/.tunnel_token"
echo "$TUNNEL_TOKEN" > "$TOKEN_FILE"
chmod 600 "$TOKEN_FILE"
echo "Saved to $TOKEN_FILE"

echo
echo "=== Setup complete ==="
echo "Tunnel ID: $TUNNEL_ID"
echo "Hostname: https://$HOSTNAME"
echo
echo "Next: deploy to fada-1 with:"
echo "  TOKEN=\$(cat $TOKEN_FILE) && experiments/rccg/r.sh play deploy_tunnel --limit fada-1 -e tunnel_token=\$TOKEN"
