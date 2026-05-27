#!/bin/bash
set -e

while [[ $# -gt 0 ]]; do
  case $1 in
    --target) TARGET="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --password) PASS="$2"; shift 2;;
    --attack-list) ATTACK_LIST="$2"; shift 2;;
    --report-xml) REPORT="$2"; shift 2;;
    *) echo "Unknown option"; exit 1;;
  esac
done

# Copy attack list to target
scp "$ATTACK_LIST" "$USER@$TARGET:/tmp/attacks.txt"

# Execute Atomic tests remotely (requires sshpass or key)
sshpass -p "$PASS" ssh "$USER@$TARGET" '
  # Install Invoke-Atomic if not present (for Linux you need powershell)
  if ! command -v pwsh &> /dev/null; then
    sudo apt-get update && sudo apt-get install -y powershell
  fi
  pwsh -c "
    Install-Module -Name AtomicRedTeam -Force -AcceptLicense -Scope CurrentUser
    Import-Module AtomicRedTeam
    \$tests = Get-Content /tmp/attacks.txt
    \$results = @()
    foreach (\$t in \$tests) {
      \$result = Invoke-AtomicTest -AtomicTechnique \$t -TestNames all -ShowDetails
      \$results += \$result
    }
    \$results | Export-JUnitReport -Path /tmp/atomic-report.xml
  "
'

# Copy report back
scp "$USER@$TARGET:/tmp/atomic-report.xml" "$REPORT"