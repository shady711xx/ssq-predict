# ==============================================================================
# 双色球自动化日常运行 Windows 定时任务注册脚本 (SSQ Daily Runner Task Registration)
#
# 运行说明:
# 1. 任务计划于 每周一、周三、周五 上午 08:02 自动触发 (双色球周日、周二、周四晚开奖)
# 2. 依次执行: 增量拉取最新开奖 -> 自动核对历史预测 -> 生成当期防撞号预测号码
# 3. 运行日志保存至: logs/daily_run.log
# 4. 最新报告生成至: reports/latest_run.md
#
# 使用方式:
# - 以管理员身份打开 PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\scripts\register_task.ps1
#
# 手动触发测试:
#   Start-ScheduledTask -TaskName "SSQ_Daily_Runner"
#
# 卸载任务:
#   Unregister-ScheduledTask -TaskName "SSQ_Daily_Runner" -Confirm:$false
# ==============================================================================

$TaskName = "SSQ_Daily_Runner"
$ProjectDir = Split-Path -Parent $PSScriptRoot
$PythonExe = (Get-Command python).Source
$LogDir = Join-Path $ProjectDir "logs"
$LogFile = Join-Path $LogDir "daily_run.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  注册双色球日常自动流水线计划任务: $TaskName" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "项目路径 : $ProjectDir"
Write-Host "Python路径: $PythonExe"
Write-Host "日志输出 : $LogFile"

# 1. Action: 运行 PowerShell 执行 python main.py run-daily 并输出日志
$ActionArg = "-NoProfile -ExecutionPolicy Bypass -Command `"Set-Location '$ProjectDir'; & '$PythonExe' main.py run-daily *>> '$LogFile'`""
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $ActionArg -WorkingDirectory $ProjectDir

# 2. Trigger: 每周一、周三、周五 08:02
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Wednesday, Friday -At 8:02AM

# 3. Settings: 允许按需运行，电池模式下也运行，失败重试
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# 4. Register or update task
try {
    # Check if task already exists
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        Write-Host "发现已有任务，正在重新注册更新..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "双色球开奖数据增量更新、自动核对与当期号码预测流水线" | Out-Null
    Write-Host "`n✅ 定时任务注册成功！" -ForegroundColor Green
    Write-Host "   • 任务名称: $TaskName"
    Write-Host "   • 调度频率: 每周一、周三、周五 08:02"
    Write-Host "   • 测试执行命令: Start-ScheduledTask -TaskName `"$TaskName`""
    Write-Host "   • 卸载任务命令: Unregister-ScheduledTask -TaskName `"$TaskName`" -Confirm:`$false`n"
} catch {
    Write-Host "`n❌ 注册失败: $_" -ForegroundColor Red
    Write-Host "请确保以管理员身份运行 PowerShell。" -ForegroundColor Yellow
}
