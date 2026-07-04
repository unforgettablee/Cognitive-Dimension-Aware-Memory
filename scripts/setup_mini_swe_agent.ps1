# Manage mini-swe-agent copies for task Docker builds
#   setup   - copy from template to all task directories (before docker build)
#   cleanup - remove from all task directories (after docker build, before git commit)
#
# Usage:
#   .\scripts\setup_mini_swe_agent.ps1 setup
#   .\scripts\setup_mini_swe_agent.ps1 cleanup

param(
    [ValidateSet("setup", "cleanup")]
    [string]$Action = "setup"
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Template = Join-Path $ProjectRoot "harbor\adapters\swebench\template\mini-swe-agent"
$TasksDir = Join-Path $ProjectRoot "harbor-tasks\swebench-verified"

if (-not (Test-Path $Template)) {
    Write-Error "Template not found at $Template"
    exit 1
}

if (-not (Test-Path $TasksDir)) {
    Write-Error "Tasks directory not found at $TasksDir"
    exit 1
}

# Font style helpers
function Write-Step {
    param([string]$Label, [string]$Status, [string]$Color = "Gray")
    Write-Host "  [$Label] " -NoNewline
    Write-Host "$Status" -ForegroundColor $Color
}

switch ($Action) {
    "setup" {
        $allTasks = @(Get-ChildItem -Path $TasksDir -Directory | ForEach-Object {
            $dst = Join-Path $_.FullName "environment\mini-swe-agent"
            if (-not (Test-Path $dst)) { $_ }
        })
        $total = $allTasks.Count
        if ($total -eq 0) {
            Write-Host "All mini-swe-agent directories already exist. Nothing to do." -ForegroundColor Green
            exit 0
        }
        Write-Host "Creating mini-swe-agent directories: $total tasks" -ForegroundColor Cyan
        $i = 0
        foreach ($task in $allTasks) {
            $i++
            $taskName = $task.Name
            Write-Progress -Activity "Copying mini-swe-agent" -Status "$taskName" -PercentComplete (($i / $total) * 100)
            Write-Step "$i/$total" $taskName "White"
            $dst = Join-Path $task.FullName "environment\mini-swe-agent"
            New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
            Copy-Item -Path $Template -Destination $dst -Recurse -ErrorAction Stop
        }
        Write-Progress -Activity "Copying mini-swe-agent" -Completed
        Write-Host "Done. Created $total mini-swe-agent directories." -ForegroundColor Green
    }
    "cleanup" {
        $allTasks = @(Get-ChildItem -Path $TasksDir -Directory | ForEach-Object {
            $dst = Join-Path $_.FullName "environment\mini-swe-agent"
            if (Test-Path $dst) { $_ }
        })
        $total = $allTasks.Count
        if ($total -eq 0) {
            Write-Host "No mini-swe-agent directories to remove." -ForegroundColor Green
            exit 0
        }
        Write-Host "Removing mini-swe-agent directories: $total tasks" -ForegroundColor Cyan
        $i = 0
        foreach ($task in $allTasks) {
            $i++
            $taskName = $task.Name
            Write-Progress -Activity "Removing mini-swe-agent" -Status "$taskName" -PercentComplete (($i / $total) * 100)
            Write-Step "$i/$total" $taskName "DarkYellow"
            $dst = Join-Path $task.FullName "environment\mini-swe-agent"
            Remove-Item -Path $dst -Recurse -Force -ErrorAction Stop
        }
        Write-Progress -Activity "Removing mini-swe-agent" -Completed
        Write-Host "Done. Removed $total mini-swe-agent directories." -ForegroundColor Green
    }
}
