#!/bin/bash
# export_logs.sh — Bundle and upload all logs from runpod_tests/logs/
# Usage: ./export_logs.sh
#
# Methods (tries in order, falls back if one fails):
#   1. transfer.sh (free, no auth, 14 days)
#   2. file.io (free, no auth, expires after download)
#   3. 0x0.st (free, no auth)
#
# Output: prints URLs for each log + a bundled tar.gz

set -e
cd "$(dirname "$0")"

LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ] || [ -z "$(ls -A $LOG_DIR 2>/dev/null)" ]; then
    echo "✗ No logs found in $LOG_DIR"
    echo "  Run setup.sh / validate.sh / unknown.sh first"
    exit 1
fi

# Bundle everything in logs/ into a tar.gz
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
HOSTNAME=$(hostname | cut -c1-8)
BUNDLE="paramgolf_logs_${HOSTNAME}_${TIMESTAMP}.tar.gz"

echo "================================================================================"
echo "EXPORT LOGS"
echo "================================================================================"
echo
echo "Creating bundle: $BUNDLE"
tar -czf "/tmp/$BUNDLE" -C "$LOG_DIR" .
SIZE=$(du -h "/tmp/$BUNDLE" | cut -f1)
echo "  Size: $SIZE"
echo

# Try upload services in order
upload_transfer_sh() {
    local file=$1
    local url
    url=$(curl --silent --max-time 60 --upload-file "$file" "https://transfer.sh/$(basename $file)" 2>/dev/null)
    if [ -n "$url" ] && [[ "$url" == https://* ]]; then
        echo "$url"
        return 0
    fi
    return 1
}

upload_file_io() {
    local file=$1
    local response
    response=$(curl --silent --max-time 60 -F "file=@$file" "https://file.io/" 2>/dev/null)
    local url=$(echo "$response" | grep -oE '"link":"[^"]+"' | sed 's/"link":"//;s/"//')
    if [ -n "$url" ]; then
        echo "$url"
        return 0
    fi
    return 1
}

upload_0x0_st() {
    local file=$1
    local url
    url=$(curl --silent --max-time 60 -F "file=@$file" "https://0x0.st" 2>/dev/null)
    if [ -n "$url" ] && [[ "$url" == http* ]]; then
        echo "$url"
        return 0
    fi
    return 1
}

# Upload bundle
echo "Uploading bundle..."
URL=""
for method in upload_transfer_sh upload_0x0_st upload_file_io; do
    echo "  trying $method..."
    URL=$($method "/tmp/$BUNDLE")
    if [ -n "$URL" ]; then
        echo "  ✓ uploaded via $method"
        break
    fi
done

echo
echo "================================================================================"
if [ -n "$URL" ]; then
    echo "✓ BUNDLE UPLOADED"
    echo
    echo "  $URL"
    echo
    echo "Download with:"
    echo "  curl -O $URL"
    echo "  tar -xzf $BUNDLE"
else
    echo "✗ ALL UPLOADS FAILED"
    echo
    echo "Bundle is at: /tmp/$BUNDLE"
    echo
    echo "Manual download options:"
    echo "  1. SCP:   scp PODHOST:/tmp/$BUNDLE ./"
    echo "  2. cat the bundle and base64-encode it for paste:"
    echo "     base64 /tmp/$BUNDLE | head -50"
fi
echo "================================================================================"

# Also upload individual logs for quick access
echo
echo "Individual log URLs:"
for log in "$LOG_DIR"/*.log; do
    if [ -f "$log" ]; then
        name=$(basename "$log")
        size=$(du -h "$log" | cut -f1)
        echo
        echo "  $name ($size):"
        URL=""
        for method in upload_transfer_sh upload_0x0_st; do
            URL=$($method "$log")
            if [ -n "$URL" ]; then
                echo "    $URL"
                break
            fi
        done
        if [ -z "$URL" ]; then
            echo "    (upload failed — see bundle)"
        fi
    fi
done

echo
echo "================================================================================"
echo "TIP: Save these URLs! Some services delete files after first download."
echo "================================================================================"
