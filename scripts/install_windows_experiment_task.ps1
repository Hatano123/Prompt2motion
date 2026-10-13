param(
    [Parameter(Mandatory = $true)][string]$Remote,
    [string]$Destination = 'C:\Users\chibi\Research\result',
    [string]$RemoteRoot = '/home/home0/y263357/Prompt2Motion',
    [string]$TaskName = 'Prompt2Motion-Experiments',
    [string]$PythonExecutable = ''
)
$ErrorActionPreference = 'Stop'
$Controller = Join-Path $PSScriptRoot 'windows_experiment_controller.py'
if (-not (Test-Path $Controller)) { throw "Missing controller: $Controller" }
$PythonArgs = @()
if ($PythonExecutable) {
    $Python = (Get-Command $PythonExecutable -ErrorAction Stop).Source
} else {
    $Python = (Get-Command py -ErrorAction SilentlyContinue).Source
    if ($Python) { $PythonArgs = @('-3') }
    else { $Python = (Get-Command python -ErrorAction Stop).Source }
}
& $Python @PythonArgs -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or later is required"; print(sys.version)'
if ($LASTEXITCODE -ne 0) { throw 'Python 3.10 or later is required. Pass -PythonExecutable with the full path to python.exe.' }
# Resolve launchers (including py.exe) to the interpreter so stopping the task
# also stops the controller rather than leaving a child process running.
$Python = (& $Python @PythonArgs -c 'import sys; print(sys.executable)' | Select-Object -Last 1)
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $Python)) { throw 'Cannot resolve Python interpreter' }
$PythonArgs = @()
& ssh -T -o BatchMode=yes -o ConnectTimeout=15 $Remote true
if ($LASTEXITCODE -ne 0) { throw 'SSH key authentication failed' }
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$ServiceDir = Join-Path $Destination '.controller'
New-Item -ItemType Directory -Force -Path $ServiceDir | Out-Null
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask -and $ExistingTask.State -eq 'Running') {
    Stop-ScheduledTask -TaskName $TaskName
}
$InstalledController = Join-Path $ServiceDir 'windows_experiment_controller.py'
Copy-Item $Controller $InstalledController -Force
@{
    Python = $Python
    PythonArgs = $PythonArgs
    Controller = $InstalledController
    Remote = $Remote
    RemoteRoot = $RemoteRoot
    Destination = $Destination
} | ConvertTo-Json | Set-Content (Join-Path $ServiceDir 'config.json') -Encoding UTF8
$Log = Join-Path $ServiceDir 'controller.log'
$ControllerArgs = @('-u', $InstalledController, '--remote', $Remote,
    '--remote-root', $RemoteRoot, '--destination', $Destination, '--loop', '--log-file', $Log)
# These paths have no embedded quotes; reject them instead of constructing an
# ambiguous Windows command line. Double trailing backslashes inside quotes.
$QuotedArgs = foreach ($Value in $ControllerArgs) {
    if ($Value.Contains('"')) { throw 'Task arguments cannot contain double quotes' }
    '"' + ($Value -replace '(\\+)$', '$1$1') + '"'
}
$UserId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$Action = New-ScheduledTaskAction -Execute $Python -Argument ($QuotedArgs -join ' ')
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId
$Principal = New-ScheduledTaskPrincipal -UserId $UserId -LogonType Interactive -RunLevel Limited
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger `
    -Principal $Principal -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Write-Host "Started $TaskName. Log: $(Join-Path $ServiceDir 'controller.log')"
Write-Host 'The controller resumes at login. Keep Windows powered on and logged in.'
