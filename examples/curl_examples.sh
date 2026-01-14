cat > examples/curl_examples.sh << 'EOF'
#!/usr/bin/env bash
set -e

echo "Health:"
curl -s http://localhost:5000/health | jq .

echo
echo "QA:"
curl -s -X POST http://localhost:5000/qa \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the main obligations."}' | jq .
EOF

chmod +x examples/curl_examples.sh
