<#
.SYNOPSIS
    Install prebuilt llama.cpp binaries so LeanLM can run a real model on Windows.

.DESCRIPTION
    The counterpart of scripts/provision_ubuntu.sh. It exists because the
    in-process route is the worst one on Windows: `llama-cpp-python` compiles
    from source, needs a C++ toolchain, and its source archive exceeds the
    260-character path limit unless long paths are enabled.

    None of that is necessary. The llama.cpp project publishes prebuilt Windows
    binaries, and LeanLM drives them directly through its `llama-cli` and
    `llama-server` backends. No compiler, no Python binding.

    This script fetches the latest release, extracts it, and tells you what to
    add to PATH. It picks a CPU x64 build: your machine has integrated graphics,
    so a CUDA or Vulkan build would be a larger download that changes nothing.

.PARAMETER Destination
    Where to extract. Defaults to tools\llama.cpp under the repository.

.PARAMETER Release
    A specific release tag. Defaults to the latest.

.EXAMPLE
    .\scripts\provision_windows.ps1
#>
[CmdletBinding()]
param(
    [string]$Destination = "tools\llama.cpp",
    [string]$Release = "",
    [string]$AssetUrl = "",
    [string]$Token = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Windows PowerShell 5.1 does not always negotiate TLS 1.2, which GitHub requires.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Show-ManualRoute {
    param([string]$Because)
    Write-Host ""
    Write-Host "  $Because" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  Download it by hand instead -- it takes a minute and always works:"
    Write-Host "    1. open  https://github.com/ggml-org/llama.cpp/releases/latest"
    Write-Host "    2. under Assets, take the Windows x64 CPU build"
    Write-Host "       (a .zip with 'win' and 'x64' in the name; skip cuda, vulkan, arm64)"
    Write-Host "    3. re-run this script with the direct link:"
    Write-Host "         .\scripts\provision_windows.ps1 -AssetUrl `"<link>`""
    Write-Host ""
    Write-Host "  Or, if the API refused you because of its rate limit, pass a token:"
    Write-Host "         .\scripts\provision_windows.ps1 -Token `"<github token>`""
}

if ($AssetUrl) {
    # The escape hatch: no API call at all.
    $assetName = [System.IO.Path]::GetFileName(($AssetUrl -split "[?#]")[0])
    $asset = [pscustomobject]@{
        name = $assetName; browser_download_url = $AssetUrl; size = 0
    }
    Write-Host "  asset  : $assetName (from -AssetUrl)"
}
else {

$api = if ($Release) {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Release"
} else {
    "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
}

Write-Host "  querying $api"
$headers = @{ "User-Agent" = "leanlm"; "Accept" = "application/vnd.github+json" }
if ($Token) { $headers["Authorization"] = "Bearer $Token" }

$release = $null
try {
    $release = Invoke-RestMethod -Uri $api -Headers $headers
}
catch {
    Show-ManualRoute "the GitHub API call failed: $($_.Exception.Message)"
    exit 1
}

# A response can arrive without being usable: an unauthenticated rate limit, a
# proxy returning an HTML page, a renamed repository. Distinguishing that from
# "this release has no matching asset" matters, because the two have different
# fixes and the first version of this script reported one as the other.
if (-not $release -or -not $release.tag_name) {
    $because = "the API responded, but not with a release."
    if ($release -and $release.message) {
        $because = "the API refused: $($release.message)"
    }
    elseif ($release -is [string]) {
        $snippet = $release.Substring(0, [Math]::Min(120, $release.Length))
        $because = "the API returned something that is not JSON (a proxy or captive portal?): $snippet"
    }
    Show-ManualRoute $because
    exit 1
}

Write-Host "  release: $($release.tag_name) ($($release.assets.Count) assets)"

# Asset names change between releases, so match on shape rather than on a fixed
# string, and exclude the accelerator builds this hardware cannot use.
$candidates = $release.assets |
    Where-Object { $_.name -match "win" -and $_.name -match "x64" -and $_.name -like "*.zip" } |
    Where-Object { $_.name -notmatch "cuda|hip|rocm|sycl|vulkan|arm64|kompute" }

# Prefer an explicit CPU or AVX2 build when the release publishes several.
$asset = $candidates | Where-Object { $_.name -match "cpu|avx2" } | Select-Object -First 1
if (-not $asset) { $asset = $candidates | Select-Object -First 1 }

if (-not $asset) {
    Write-Host "  no Windows x64 CPU asset in $($release.tag_name). Published assets:" -ForegroundColor Red
    $release.assets | ForEach-Object { Write-Host "    $($_.name)" }
    Show-ManualRoute "none of them matched a Windows x64 CPU build."
    exit 1
}

Write-Host "  asset  : $($asset.name) ($([math]::Round($asset.size / 1MB, 0)) MB)"

}   # end of the API branch

$temp = Join-Path ([System.IO.Path]::GetTempPath()) $asset.name
$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if ($curl) {
    & $curl.Source -L --fail --progress-bar -o $temp $asset.browser_download_url
    if ($LASTEXITCODE -ne 0) { Write-Error "download failed"; exit 1 }
}
else {
    $client = New-Object System.Net.WebClient
    try { $client.DownloadFile($asset.browser_download_url, $temp) } finally { $client.Dispose() }
}

# Absolute path: .NET resolves relative paths against the process working
# directory, not $PWD.
$Destination = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine($PWD.ProviderPath, $Destination))
if (Test-Path $Destination) { Remove-Item $Destination -Recurse -Force }
New-Item -ItemType Directory -Path $Destination -Force | Out-Null

Write-Host "  extracting to $Destination"
Expand-Archive -Path $temp -DestinationPath $Destination -Force
Remove-Item $temp -Force

# Some releases nest the binaries one level down.
$bench = Get-ChildItem -Path $Destination -Filter "llama-bench.exe" -Recurse |
    Select-Object -First 1
if (-not $bench) {
    Write-Host "error: the archive contains no llama-bench.exe" -ForegroundColor Red
    Get-ChildItem -Path $Destination -Recurse -Filter "*.exe" |
        ForEach-Object { Write-Host "  found: $($_.Name)" }
    exit 1
}
$binDir = $bench.Directory.FullName

Write-Host ""
Write-Host "  binaries in $binDir"
foreach ($name in @("llama-bench.exe", "llama-cli.exe", "llama-server.exe")) {
    $present = Test-Path (Join-Path $binDir $name)
    Write-Host ("    {0,-20} {1}" -f $name, $(if ($present) { "yes" } else { "MISSING" }))
}

Write-Host ""
Write-Host "  add to PATH for this session:"
Write-Host "    `$env:PATH = `"$binDir;`$env:PATH`""
Write-Host ""
Write-Host "  permanently, for your user:"
Write-Host "    [Environment]::SetEnvironmentVariable('PATH', `"$binDir;`" + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')"
Write-Host ""
Write-Host "  then:"
Write-Host "    leanlm doctor"
Write-Host "    leanlm --backend llama-cli ask `"What is the reimbursement deadline for expense reports?`""
Write-Host ""
Write-Host "  note: these binaries measure and generate. They do not make this"
Write-Host "        machine a conforming one -- the submitted numbers still have to"
Write-Host "        come from hardware close to the 4 vCPU / 8 GB profile."
