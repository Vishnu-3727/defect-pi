param([string]$PiArg = "")
$PiUser = "vishnu"
$PiHost = "defect-pi.local"
$Port = 5000
$ForceLocal = $false
$ExplicitHost = $false
if ($PiArg -ne "") {
  if ($PiArg -eq "local") { $ForceLocal = $true } else { $PiHost = $PiArg; $ExplicitHost = $true }
} elseif ($env:PI_HOST -ne $null -and $env:PI_HOST -ne "") {
  $PiHost = $env:PI_HOST
  $ExplicitHost = $true
}
$ScriptDir = $PSScriptRoot
$LastIpFile = Join-Path $ScriptDir ".pi_last_ip"
function Test-PiReachable {
  param([string]$HostName)
  $client = New-Object System.Net.Sockets.TcpClient
  try {
    $task = $client.ConnectAsync($HostName, 22)
    if ($task.Wait(3000)) { return $client.Connected }
    return $false
  } catch { return $false } finally { $client.Close() }
}
function Test-Http200 {
  param([string]$Url)
  try {
    $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
    if ($r.StatusCode -eq 200) { return $true }
    return $false
  } catch { return $false }
}
function Wait-Http200 {
  param([string]$Url, [int]$TimeoutSec)
  for ($i = 0; $i -lt $TimeoutSec; $i++) {
    if (Test-Http200 $Url) { return $true }
    Start-Sleep -Seconds 1
  }
  return $false
}
function Start-LaptopMode {
  if ((Get-Command uv -ErrorAction SilentlyContinue) -eq $null) {
    Write-Host 'uv not found. Install it with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
  }
  Set-Location $ScriptDir
  $Url = "http://localhost:$Port/"
  Start-Job -ScriptBlock {
    param([string]$WaitUrl)
    for ($i = 0; $i -lt 60; $i++) {
      try {
        $r = Invoke-WebRequest -Uri $WaitUrl -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { Start-Process $WaitUrl; break }
      } catch { }
      Start-Sleep -Seconds 1
    }
  } -ArgumentList $Url | Out-Null
  uv run python dashboard/app.py --source bench_fixtures/images --loop
}
$FoundHost = ""
$TriedHosts = @()
if ($ForceLocal -eq $false) {
  if ($ExplicitHost) {
    $TriedHosts = @($PiHost)
  } else {
    $TriedHosts = @("defect-pi.local")
    if (Test-Path $LastIpFile) {
      try {
        $saved = Get-Content $LastIpFile -TotalCount 1 | Select-Object -First 1
        if ($saved -ne $null) {
          $saved = $saved.Trim()
          if (($saved -ne "") -and ($saved -ne "defect-pi.local")) { $TriedHosts += $saved }
        }
      } catch { }
    }
  }
  foreach ($h in $TriedHosts) {
    if (Test-PiReachable $h) { $FoundHost = $h; break }
  }
  if ($FoundHost -ne "") {
    $PiHost = $FoundHost
    try {
      $ip = [System.Net.Dns]::GetHostAddresses($FoundHost) | Where-Object AddressFamily -eq 'InterNetwork' | Select-Object -First 1
      if ($ip -ne $null) { $ip.ToString() | Set-Content $LastIpFile }
    } catch { }
  }
}
if (($ForceLocal -eq $false) -and ($FoundHost -ne "")) {
  $PiUrl = "http://${PiHost}:${Port}/"
  if (Test-Http200 $PiUrl) {
    Write-Host "Dashboard already running on Pi: $PiUrl"
    Start-Process $PiUrl
    exit 0
  }
  ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$PiUser@$PiHost" demo start defect-inspect
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ssh to $PiHost failed (exit $LASTEXITCODE). Falling back to laptop mode."
    Start-LaptopMode
    exit $LASTEXITCODE
  }
  if (Wait-Http200 $PiUrl 30) { Start-Process $PiUrl } else { Write-Host "Pi dashboard did not answer at $PiUrl within 30 s." }
  Write-Host "To stop the Pi demo: ssh $PiUser@$PiHost demo stop"
  exit 0
}
if ($ForceLocal -eq $false -and $FoundHost -eq "") {
  $tried = $TriedHosts -join ", "
  Write-Host "Pi not found (tried: $tried). Using laptop mode."
}
Start-LaptopMode
