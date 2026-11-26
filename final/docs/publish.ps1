# Run locally with the published checksums. The -Publish switch enables the remote push.
[CmdletBinding()]
param(
    [ValidateSet('Bundle','Zip')][string]$Method = 'Bundle',
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-f]{40}$')][string]$FinalCommit,
    [Parameter(Mandatory=$true)][ValidatePattern('^[0-9a-fA-F]{64}$')][string]$FileSHA256,
    [string]$DownloadDirectory = (Join-Path $env:USERPROFILE 'Downloads'),
    [string]$Repository = 'C:\Users\saipo\Desktop\sem_5\Digital Image Processing\project\unrippling',
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
function CheckedGit {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "Git failed. Review the command and repository state before continuing." }
}
function VerifyRemote {
    $allowed = @('https://github.com/rtvk2/unrippling.git','https://github.com/rtvk2/unrippling','git@github.com:rtvk2/unrippling.git')
    foreach ($url in @((CheckedGit remote get-url --all origin); (CheckedGit remote get-url --push --all origin))) {
        if ($allowed -notcontains $url) { throw "Unexpected origin URL: $url" }
    }
}
Set-Location -LiteralPath $Repository
if (CheckedGit status --porcelain) { throw 'Working tree has changes. Preserve them before delivery.' }
VerifyRemote
$base = '792549e00b02e4a1a71f8d26c0d7d15d933300dd'
$branch = 'complete-unrippling'
CheckedGit switch $branch
CheckedGit fetch origin
CheckedGit merge --ff-only "origin/$branch"
if ((CheckedGit rev-parse HEAD) -ne $base) {
    throw "Expected clean consolidation base $base. Compare local and remote history before applying the delivery."
}
$filename = if ($Method -eq 'Bundle') { 'unrippling-final.bundle' } else { 'unrippling-final-project.zip' }
$file = Join-Path $DownloadDirectory $filename
if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $FileSHA256) { throw 'Download checksum mismatch.' }

if ($Method -eq 'Bundle') {
    CheckedGit bundle verify $file
    CheckedGit bundle list-heads $file
    CheckedGit fetch $file refs/heads/final-delivery
    if ((CheckedGit rev-parse FETCH_HEAD) -ne $FinalCommit) { throw 'Imported a different commit.' }
    CheckedGit merge-base --is-ancestor $base $FinalCommit
    CheckedGit merge --ff-only $FinalCommit
    $expected = $FinalCommit
} else {
    $temp = Join-Path ([IO.Path]::GetTempPath()) ("unrippling-" + [guid]::NewGuid())
    Expand-Archive -LiteralPath $file -DestinationPath $temp
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $temp 'delivery-manifest.json') | ConvertFrom-Json
    if ($manifest.base -ne $base -or $manifest.commit -ne $FinalCommit) { throw 'Unexpected ZIP manifest.' }
    if ($manifest.removals.Count -ne 0) { throw 'Unexpected removals in ZIP delivery.' }
    # Validate every destination before writing any file. The manifest contains only added/modified paths.
    foreach ($entry in $manifest.files) {
        $p = [string]$entry.path
        if ($p -notmatch '^(final/|README\.md$|\.gitignore$)' -or $p -match '(^|/)\.\.?(/|$)|[:\\]') { throw "Unexpected path: $p" }
        if ($entry.status -eq 'A' -and (Test-Path -LiteralPath $p)) { throw "Existing local file would be replaced: $p" }
        $src = Join-Path $temp $p
        if ((CheckedGit hash-object --no-filters -- $src) -ne $entry.blob) { throw "Blob mismatch: $p" }
    }
    foreach ($entry in $manifest.files) {
        $p = [string]$entry.path
        $parent = Split-Path $p -Parent
        if ($parent) { [IO.Directory]::CreateDirectory((Join-Path $Repository $parent)) | Out-Null }
        Copy-Item -LiteralPath (Join-Path $temp $p) -Destination $p -Force
        CheckedGit -c core.autocrlf=false add -- $p
    }
    if ((CheckedGit write-tree) -ne $manifest.tree) { throw 'Staged tree mismatch. Inspect git diff --cached.' }
    CheckedGit commit -m 'Finalize Un-Rippling course project from verified ZIP'
    $expected = CheckedGit rev-parse HEAD
    CheckedGit merge-base --is-ancestor $base $expected
    Write-Host "ZIP produced a local commit with the verified delivery tree. Temporary copy: $temp"
}
if ((CheckedGit rev-parse HEAD) -ne $expected) { throw 'Unexpected local tip.' }
foreach ($n in 1..5) {
    foreach ($p in @("final/stage_$n/run_stage_$n.py","final/stage_$n/test_stage_$n.py","final/stage_$n/results")) {
        if (!(Test-Path -LiteralPath $p)) { throw "Missing $p" }
    }
}
if (!(Test-Path final/docs/project_report.pdf)) { throw 'Report missing.' }
CheckedGit status --short --branch
CheckedGit log -1 --oneline
if ($Publish) {
    VerifyRemote
    CheckedGit fetch origin
    CheckedGit merge-base --is-ancestor "origin/$branch" $expected
    CheckedGit push origin "HEAD:refs/heads/$branch"
    CheckedGit fetch origin
    CheckedGit merge-base --is-ancestor $expected "origin/$branch"
    CheckedGit ls-remote --heads origin "refs/heads/$branch"
}
