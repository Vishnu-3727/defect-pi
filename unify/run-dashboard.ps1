param(
  [string]$PiArg = ""
)

# Run the UNIFY-RUL dashboard, on the Pi if it answers, else on this laptop.
#   run-dashboard.ps1 [host|local]   ($PI_HOST works the same as the argument)
# Pi user is vishnu. TCP 22 within 3 s decides whether the Pi is there.

$ScriptDir = $PSScriptRoot
if ($ScriptDir -eq $null -or $ScriptDir -eq "") {
  $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$LastIpFile = Join-Path $ScriptDir ".pi_last_ip"
$PiUser = "vishnu"

function Test-Tcp22 {
  param([string]$H)
  try {
    $c = New-Object System.Net.Sockets.TcpClient
    $iar = $c.BeginConnect($H, 22, $null, $null)
    $ok = $iar.AsyncWaitHandle.WaitOne(3000)
    if (-not $ok) {
      $c.Close()
      return $false
    }
    $connected = $c.Connected
    $c.Close()
    return $connected
  } catch {
    return $false
  }
}

function Test-Health {
  param([string]$Url)
  try {
    $r = Invoke-RestMethod -Uri $Url -TimeoutSec 3
    if ($r.PSObject.Properties.Name -contains 'features') {
      return $true
    }
    return $false
  } catch {
    return $false
  }
}

function Save-PiIp {
  param([string]$H)
  try {
    $addrs = [System.Net.Dns]::GetHostAddresses($H)
    foreach ($a in $addrs) {
      if ($a.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork) {
        Set-Content -LiteralPath $LastIpFile -Value $a.IPAddressToString
        break
      }
    }
  } catch {
  }
}

function Get-FreePort {
  $candidates = @(8000, 8010, 8020, 8030)
  foreach ($p in $candidates) {
    try {
      $l = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Parse("127.0.0.1"), $p)
      $l.Start()
      $l.Stop()
      return $p
    } catch {
    }
  }
  return $null
}

function Start-LaptopMode {
  if ((Get-Command uv -ErrorAction SilentlyContinue) -eq $null) {
    Write-Host 'uv not found. Install it with: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    exit 1
  }
  $Port = 0
  if ($env:UNIFY_PORT -ne $null -and $env:UNIFY_PORT -ne "") {
    $Port = [int]$env:UNIFY_PORT
  } else {
    $Port = Get-FreePort
    if ($Port -eq $null) {
      Write-Host "No free port among 8000, 8010, 8020, 8030."
      exit 1
    }
  }
  Write-Host "Using port $Port."
  $EdgeDir = Join-Path $ScriptDir "edge"
  Write-Host "Starting dashboard on this laptop..."
  $server = Start-Process -PassThru -NoNewWindow -WorkingDirectory $EdgeDir -FilePath "uv" -ArgumentList "run","--with-requirements","requirements-pi.txt","python","-m","uvicorn","app:app","--host","127.0.0.1","--port","$Port"
  $ready = $false
  for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 1
    if ($server.HasExited) {
      Write-Host "UNIFY server exited early"
      exit 1
    }
    if (Test-Health "http://127.0.0.1:${Port}/health") {
      $ready = $true
      break
    }
  }
  if (-not $ready) {
    Write-Host "Server did not answer /health within 90 s."
    try { & taskkill /PID $server.Id /T /F } catch { }
    exit 1
  }
  $sim = Start-Process -PassThru -NoNewWindow -WorkingDirectory $EdgeDir -FilePath "uv" -ArgumentList "run","--with-requirements","requirements-pi.txt","python","fake_node.py","http://127.0.0.1:${Port}/ingest","--degrade","--loop"
  Start-Process "http://127.0.0.1:${Port}/"
  Write-Host "Press Enter (Windows) / Ctrl+C (mac) to stop"
  try {
    Read-Host
  } finally {
    try { & taskkill /PID $server.Id /T /F } catch { }
    try { & taskkill /PID $sim.Id /T /F } catch { }
  }
}

$ForceLocal = $false
if ($PiArg -eq "local") {
  $ForceLocal = $true
}

$OnlyCandidate = ""
if ($PiArg -ne "" -and $PiArg -ne "local") {
  $OnlyCandidate = $PiArg
} else {
  if ($env:PI_HOST -ne $null) {
    if ($env:PI_HOST -ne "") {
      $OnlyCandidate = $env:PI_HOST
    }
  }
}

$PiHost = ""
if (-not $ForceLocal) {
  if ($OnlyCandidate -ne "") {
    if (Test-Tcp22 $OnlyCandidate) {
      $PiHost = $OnlyCandidate
    }
  } else {
    if (Test-Tcp22 "defect-pi.local") {
      $PiHost = "defect-pi.local"
    } else {
      if (Test-Path -LiteralPath $LastIpFile) {
        $saved = (Get-Content -LiteralPath $LastIpFile | Select-Object -First 1).Trim()
        if ($saved -ne "" -and $saved -ne $null) {
          if (Test-Tcp22 $saved) {
            $PiHost = $saved
          }
        }
      }
    }
  }
}

if ($PiHost -ne "" -and -not $ForceLocal) {
  Save-PiIp $PiHost
  $healthUrl = "http://${PiHost}:8000/health"
  $pageUrl = "http://${PiHost}:8000/"
  if (Test-Health $healthUrl) {
    Write-Host "Pi already serving at ${pageUrl}"
    Start-Process $pageUrl
    Write-Host "To stop: ssh ${PiUser}@${PiHost} demo stop"
    exit 0
  }
  Write-Host "Starting demo on the Pi..."
  & ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "${PiUser}@${PiHost}" demo start unify-rul
  if ($LASTEXITCODE -ne 0) {
    Write-Host "ssh failed, falling back to laptop mode."
    Start-LaptopMode
    exit 0
  }
  $ready = $false
  for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Health $healthUrl) {
      $ready = $true
      break
    }
  }
  if (-not $ready) {
    Write-Host "Pi did not serve /health within 30 s, falling back to laptop mode."
    Start-LaptopMode
    exit 0
  }
  Start-Process $pageUrl
  Write-Host "To stop: ssh ${PiUser}@${PiHost} demo stop"
  exit 0
}

Start-LaptopMode
