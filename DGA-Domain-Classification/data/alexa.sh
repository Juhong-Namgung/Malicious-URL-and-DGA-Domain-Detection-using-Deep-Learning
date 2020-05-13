#!/bin/bash
# This script will fetch the Alexa top 1 million domains and prepare them how I like it.

# Check if unzip is installed
[ -f /usr/bin/unzip ] || sudo apt-get install unzip

# Get Alexa Top 1 mil
wget -q --show-progress http://s3.amazonaws.com/alexa-static/top-1m.csv.zip

# Unzip
unzip top-1m.csv.zip

# Parse
cat top-1m.csv | cut -d, -f2 > alexa-top-1m.txt

# Finished
echo "[+] Saved to alexa-top-1m.txt"
