param(
    [Parameter(Mandatory = $true)]
    [int]$LauncherPid,
    [int]$BackendPid = 0,
    [int]$FrontendPid = 0
)

$ErrorActionPreference = "SilentlyContinue"

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
        [int[]]$RootIds
    )

    foreach ($processId in (Get-ProcessTreeIds -RootIds $RootIds)) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

while ($true) {
    try {
        Get-Process -Id $LauncherPid -ErrorAction Stop | Out-Null
    }
    catch {
        break
    }

    Start-Sleep -Milliseconds 800
}

Stop-ProcessTreeById -RootIds @($BackendPid, $FrontendPid)
