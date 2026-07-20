param()

$ErrorActionPreference = "Stop"

function Ensure-LauncherInterop {
    if (([System.Management.Automation.PSTypeName]'LauncherInterop.JobObjectNative').Type) {
        return
    }

    Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

namespace LauncherInterop
{
    public static class JobObjectNative
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        public static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string lpName);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool SetInformationJobObject(
            IntPtr hJob,
            int jobObjectInfoClass,
            IntPtr lpJobObjectInfo,
            uint cbJobObjectInfoLength);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool CloseHandle(IntPtr hObject);
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }
}
"@
}

function Normalize-ProcessPathEnvironment {
    $pathCandidates = @()
    foreach ($key in @("Path", "PATH")) {
        $value = [System.Environment]::GetEnvironmentVariable($key, "Process")
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            $pathCandidates += $value
        }
        [System.Environment]::SetEnvironmentVariable($key, $null, "Process")
    }

    if ($pathCandidates.Count -eq 0) {
        return
    }

    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
    $segments = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in $pathCandidates) {
        foreach ($segment in ($candidate -split ';')) {
            $trimmed = $segment.Trim()
            if (-not [string]::IsNullOrWhiteSpace($trimmed) -and $seen.Add($trimmed)) {
                $segments.Add($trimmed)
            }
        }
    }

    [System.Environment]::SetEnvironmentVariable("Path", ($segments -join ';'), "Process")
}

function New-LauncherJobObject {
    Ensure-LauncherInterop

    $jobHandle = [LauncherInterop.JobObjectNative]::CreateJobObject([IntPtr]::Zero, "PLDatabaseLauncher-$PID")
    if ($jobHandle -eq [IntPtr]::Zero) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Failed to create Windows job object (Win32=$errorCode)."
    }

    $info = New-Object LauncherInterop.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    $info.BasicLimitInformation.LimitFlags = 0x2000
    $infoSize = [Runtime.InteropServices.Marshal]::SizeOf($info)
    $infoPtr = [Runtime.InteropServices.Marshal]::AllocHGlobal($infoSize)

    try {
        [Runtime.InteropServices.Marshal]::StructureToPtr($info, $infoPtr, $false)
        $ok = [LauncherInterop.JobObjectNative]::SetInformationJobObject($jobHandle, 9, $infoPtr, [uint32]$infoSize)
        if (-not $ok) {
            $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
            throw "Failed to configure Windows job object (Win32=$errorCode)."
        }
    }
    finally {
        [Runtime.InteropServices.Marshal]::FreeHGlobal($infoPtr)
    }

    return $jobHandle
}

function Add-ProcessToLauncherJob {
    param(
        [IntPtr]$JobHandle,
        [System.Diagnostics.Process]$Process
    )

    if ($null -eq $Process -or $JobHandle -eq [IntPtr]::Zero) {
        return
    }

    $ok = [LauncherInterop.JobObjectNative]::AssignProcessToJobObject($JobHandle, $Process.Handle)
    if (-not $ok) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "Failed to attach PID $($Process.Id) to launcher job object (Win32=$errorCode)."
    }
}

function Test-ProjectBackendRunning {
    param(
        [string]$ProjectRoot
    )

    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8110/health" -TimeoutSec 2
        if ($null -eq $health) {
            return $false
        }

        $configPath = [string]$health.config_path
        return ($health.status -eq "ok" -and $configPath.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase))
    }
    catch {
        return $false
    }
}

function Get-ProjectFrontendProcess {
    param(
        [string]$FrontendRoot
    )

    $frontendToken = $FrontendRoot.ToLowerInvariant()
    $viteToken = "vite"

    try {
        $nodeProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'node.exe'" -ErrorAction Stop)
    }
    catch {
        return $null
    }

    foreach ($process in $nodeProcesses) {
        $commandLine = [string]$process.CommandLine
        if ([string]::IsNullOrWhiteSpace($commandLine)) {
            continue
        }

        $normalizedCommandLine = $commandLine.ToLowerInvariant()
        if ($normalizedCommandLine.Contains($frontendToken) -and $normalizedCommandLine.Contains($viteToken)) {
            return $process
        }
    }

    return $null
}

function Get-ProjectBackendLauncherProcess {
    param(
        [string]$ProjectRoot,
        [string]$PythonExe
    )

    $projectToken = $ProjectRoot.ToLowerInvariant()
    $pythonToken = $PythonExe.ToLowerInvariant()

    try {
        $pythonProcesses = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'pythonw.exe'" -ErrorAction Stop)
    }
    catch {
        return $null
    }

    foreach ($process in $pythonProcesses) {
        $commandLine = [string]$process.CommandLine
        $executablePath = [string]$process.ExecutablePath
        if ([string]::IsNullOrWhiteSpace($commandLine)) {
            continue
        }

        $normalizedCommandLine = $commandLine.ToLowerInvariant()
        $normalizedExecutablePath = $executablePath.ToLowerInvariant()
        if (
            ($normalizedExecutablePath -eq $pythonToken -or $normalizedCommandLine.Contains($pythonToken)) -and
            $normalizedCommandLine.Contains($projectToken) -and
            $normalizedCommandLine.Contains("uvicorn") -and
            $normalizedCommandLine.Contains("backend.app:app")
        ) {
            return $process
        }
    }

    return $null
}

function Get-ListeningProcessByPort {
    param(
        [int]$Port
    )

    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
    }
    catch {
        return $null
    }

    if ($null -eq $connection) {
        return $null
    }

    try {
        return Get-CimInstance Win32_Process -Filter ("ProcessId = {0}" -f [int]$connection.OwningProcess) -ErrorAction Stop
    }
    catch {
        return $null
    }
}

function Get-ProcessTreeIds {
    param(
        [int[]]$RootIds
    )

    $uniqueRoots = @($RootIds | Where-Object { $_ -gt 0 } | Sort-Object -Unique)
    if ($uniqueRoots.Count -eq 0) {
        return @()
    }

    try {
        $allProcesses = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    }
    catch {
        return $uniqueRoots
    }

    $childrenByParent = @{}
    foreach ($process in $allProcesses) {
        $parentId = [int]$process.ParentProcessId
        if (-not $childrenByParent.ContainsKey($parentId)) {
            $childrenByParent[$parentId] = New-Object System.Collections.Generic.List[int]
        }
        $childrenByParent[$parentId].Add([int]$process.ProcessId)
    }

    $visited = New-Object 'System.Collections.Generic.HashSet[int]'
    $ordered = New-Object System.Collections.Generic.List[int]

    function Add-TreePostOrder {
        param([int]$ProcessId)

        if (-not $visited.Add($ProcessId)) {
            return
        }

        if ($childrenByParent.ContainsKey($ProcessId)) {
            foreach ($childId in $childrenByParent[$ProcessId]) {
                Add-TreePostOrder -ProcessId $childId
            }
        }

        $ordered.Add($ProcessId) | Out-Null
    }

    foreach ($rootId in $uniqueRoots) {
        Add-TreePostOrder -ProcessId $rootId
    }

    return @($ordered)
}

function Stop-ProcessTreeById {
    param(
        [int[]]$RootIds,
        [string]$Label
    )

    $treeIds = @(Get-ProcessTreeIds -RootIds $RootIds)
    if ($treeIds.Count -eq 0) {
        return
    }

    Write-Host "Stopping existing $Label processes: $($treeIds -join ', ')" -ForegroundColor DarkYellow
    foreach ($processId in $treeIds) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

function Wait-PortsReleased {
    param(
        [int[]]$Ports,
        [int]$TimeoutSeconds = 12
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $busyPorts = @()
        foreach ($port in $Ports) {
            try {
                $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop | Select-Object -First 1
            }
            catch {
                $connection = $null
            }

            if ($null -ne $connection) {
                $busyPorts += $port
            }
        }

        if ($busyPorts.Count -eq 0) {
            return $true
        }

        Start-Sleep -Milliseconds 500
    }
    while ((Get-Date) -lt $deadline)

    return $false
}

function New-LauncherSessionId {
    return "{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff"), $PID
}

function Get-LogTimestamp {
    return (Get-Date).ToUniversalTime().ToString("o")
}

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendRoot = Join-Path $projectRoot "frontend"
$npmCmd = (Get-Command npm.cmd -ErrorAction Stop).Source
$powershellExe = (Get-Command powershell.exe -ErrorAction Stop).Source
$watchdogScript = Join-Path $PSScriptRoot "launcher_watchdog.ps1"
$logRoot = Join-Path $projectRoot "data\logs\launcher"
$existingBackend = Test-ProjectBackendRunning -ProjectRoot $projectRoot
$existingFrontend = Get-ProjectFrontendProcess -FrontendRoot $frontendRoot
$existingBackendLauncher = Get-ProjectBackendLauncherProcess -ProjectRoot $projectRoot -PythonExe $pythonExe
$backendPortOwner = $null
if ($existingBackend) {
    $backendPortOwner = Get-ListeningProcessByPort -Port 8110
}

if ($existingBackend -or $null -ne $existingFrontend -or $null -ne $existingBackendLauncher -or $null -ne $backendPortOwner) {
    Write-Host ""
    Write-Host "Existing PL_database processes were detected. Cleaning them up before restart." -ForegroundColor Cyan
    Write-Host ""

    $backendRoots = @()
    if ($null -ne $existingBackendLauncher) {
        $backendRoots += [int]$existingBackendLauncher.ProcessId
    }
    if ($null -ne $backendPortOwner) {
        $backendRoots += [int]$backendPortOwner.ProcessId
    }
    $backendRoots = @($backendRoots | Sort-Object -Unique)
    if ($backendRoots.Count -gt 0) {
        Stop-ProcessTreeById -RootIds $backendRoots -Label "backend"
    }

    if ($null -ne $existingFrontend) {
        Stop-ProcessTreeById -RootIds @([int]$existingFrontend.ProcessId) -Label "frontend"
    }

    if (-not (Wait-PortsReleased -Ports @(8110, 5173) -TimeoutSeconds 12)) {
        throw "Timed out waiting for ports 8110/5173 to be released after stopping the previous launcher instance."
    }
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$sessionId = New-LauncherSessionId
$sessionLogRoot = Join-Path $logRoot $sessionId
New-Item -ItemType Directory -Force -Path $sessionLogRoot | Out-Null

try {
    [System.IO.File]::WriteAllText((Join-Path $logRoot "latest.txt"), $sessionLogRoot + [Environment]::NewLine)
}
catch {
}

$backendLog = Join-Path $sessionLogRoot "backend.log"
$backendStdoutLog = Join-Path $sessionLogRoot "backend.stdout.log"
$backendStderrLog = Join-Path $sessionLogRoot "backend.stderr.log"
$frontendLog = Join-Path $sessionLogRoot "frontend.log"
$frontendStdoutLog = Join-Path $sessionLogRoot "frontend.stdout.log"
$frontendStderrLog = Join-Path $sessionLogRoot "frontend.stderr.log"
$launcherJob = [IntPtr]::Zero

foreach ($path in @(
    $backendLog,
    $backendStdoutLog,
    $backendStderrLog,
    $frontendLog,
    $frontendStdoutLog,
    $frontendStderrLog
)) {
    New-Item -ItemType File -Force -Path $path | Out-Null
}

$backendArgs = @(
    "-m", "uvicorn",
    "backend.app:app",
    "--host", "127.0.0.1",
    "--port", "8110",
    "--no-access-log"
)

$backend = $null
$frontend = $null
$watchdog = $null
$startedProcesses = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

$logState = @{
    "backend.stdout" = 0
    "backend.stderr" = 0
    "frontend.stdout" = 0
    "frontend.stderr" = 0
}

function Write-NewLogLines {
    param(
        [string]$Key,
        [string]$Label,
        [string]$Path,
        [string]$CombinedLog
    )

    if (-not (Test-Path $Path)) {
        return
    }

    $lines = Get-Content -Path $Path -ErrorAction SilentlyContinue
    if ($null -eq $lines) {
        return
    }

    $lineCount = @($lines).Count
    if ($lineCount -le $logState[$Key]) {
        return
    }

    $startIndex = $logState[$Key]
    $endIndex = $lineCount - 1
    for ($index = $startIndex; $index -le $endIndex; $index++) {
        $line = $lines[$index]
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        $renderedLine = "[{0}] [{1}] {2}" -f (Get-LogTimestamp), $Label, $line
        Add-Content -Path $CombinedLog -Value $renderedLine
        Write-Host $renderedLine
    }

    $logState[$Key] = $lineCount
}

try {
    $launcherJob = New-LauncherJobObject

    $backend = Start-Process `
        -FilePath $pythonExe `
        -ArgumentList $backendArgs `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput $backendStdoutLog `
        -RedirectStandardError $backendStderrLog `
        -WindowStyle Hidden `
        -PassThru
    Add-ProcessToLauncherJob -JobHandle $launcherJob -Process $backend
    $startedProcesses.Add($backend) | Out-Null

    Write-Host "Waiting for backend to become ready..." -ForegroundColor DarkCyan
    $backendReadyDeadline = (Get-Date).AddSeconds(25)
    while (-not (Test-ProjectBackendRunning -ProjectRoot $projectRoot)) {
        Write-NewLogLines -Key "backend.stdout" -Label "backend" -Path $backendStdoutLog -CombinedLog $backendLog
        Write-NewLogLines -Key "backend.stderr" -Label "backend:stderr" -Path $backendStderrLog -CombinedLog $backendLog

        if ($backend.HasExited) {
            throw "Backend exited before becoming ready. Check $backendLog"
        }

        if ((Get-Date) -ge $backendReadyDeadline) {
            throw "Backend did not become ready within 25 seconds. Check $backendLog"
        }

        Start-Sleep -Milliseconds 350
    }

    $frontend = Start-Process `
        -FilePath $npmCmd `
        -ArgumentList @("run", "dev") `
        -WorkingDirectory $frontendRoot `
        -RedirectStandardOutput $frontendStdoutLog `
        -RedirectStandardError $frontendStderrLog `
        -WindowStyle Hidden `
        -PassThru
    Add-ProcessToLauncherJob -JobHandle $launcherJob -Process $frontend
    $startedProcesses.Add($frontend) | Out-Null

    $watchdog = Start-Process `
        -FilePath $powershellExe `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $watchdogScript,
            "-LauncherPid", "$PID",
            "-BackendPid", "$($backend.Id)",
            "-FrontendPid", "$($frontend.Id)"
        ) `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -PassThru

    Write-Host ""
    Write-Host "PL_database launcher started." -ForegroundColor Cyan
    Write-Host "Backend PID : $($backend.Id)" -ForegroundColor Yellow
    Write-Host "Frontend PID: $($frontend.Id)" -ForegroundColor Yellow
    Write-Host "Backend URL : http://127.0.0.1:8110" -ForegroundColor Green
    Write-Host "Frontend URL: http://127.0.0.1:5173" -ForegroundColor Green
    Write-Host "Import path : Python witio importer from config.yaml" -ForegroundColor Green
    Write-Host "Launcher logs: $sessionLogRoot" -ForegroundColor Green
    Write-Host ""
    Write-Host "Streaming combined backend/frontend logs below." -ForegroundColor Cyan
    Write-Host "Press Ctrl+C, stop the terminal, or close the terminal window to stop both processes." -ForegroundColor Cyan
    Write-Host ""

    while ($true) {
        if ($null -ne $backend) {
            Write-NewLogLines -Key "backend.stdout" -Label "backend" -Path $backendStdoutLog -CombinedLog $backendLog
            Write-NewLogLines -Key "backend.stderr" -Label "backend:stderr" -Path $backendStderrLog -CombinedLog $backendLog
        }

        if ($null -ne $frontend) {
            Write-NewLogLines -Key "frontend.stdout" -Label "frontend" -Path $frontendStdoutLog -CombinedLog $frontendLog
            Write-NewLogLines -Key "frontend.stderr" -Label "frontend:stderr" -Path $frontendStderrLog -CombinedLog $frontendLog
        }

        if ($startedProcesses.Count -eq 0) {
            break
        }

        $runningProcesses = @($startedProcesses | Where-Object { -not $_.HasExited })
        if ($runningProcesses.Count -eq 0) {
            break
        }

        Start-Sleep -Milliseconds 700
    }
}
finally {
    foreach ($process in $startedProcesses) {
        if ($null -ne $process) {
            try {
                if (-not $process.HasExited) {
                    Stop-Process -Id $process.Id -Force
                }
            }
            catch {
            }
        }
    }

    if ($launcherJob -ne [IntPtr]::Zero) {
        [LauncherInterop.JobObjectNative]::CloseHandle($launcherJob) | Out-Null
    }
}
