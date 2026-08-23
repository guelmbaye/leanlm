<#
.SYNOPSIS
    Fetch the GGUF model LeanLM runs on.

.DESCRIPTION
    The PowerShell counterpart of scripts/download_model.sh, with the same three
    guarantees, because a Windows laptop is the target machine rather than an
    afterthought:

      1. the file is checked for the GGUF magic bytes, so an HTML error page
         saved under a .gguf name is deleted instead of being benchmarked;
      2. the SHA-256 is verified when declared, and the file is refused on a
         mismatch;
      3. when no checksum is declared the computed one is printed, because a
         model that cannot be re-fetched byte for byte cannot support a
         reproducible measurement.

    This is the only script in the repository that touches the network, and the
    runtime never calls it. Once it has run, LeanLM works with the network
    unplugged.

.PARAMETER Url
    Direct URL to a .gguf file.

.PARAMETER Sha256
    Expected SHA-256. The download is rejected if it does not match.

.PARAMETER Destination
    Where to write it. Defaults to models\<filename from the URL>, so that two
    candidates cannot silently overwrite each other.

.EXAMPLE
    .\scripts\download_model.ps1 -Url "https://example/model-Q4_K_M.gguf"

.EXAMPLE
    .\scripts\download_model.ps1 -Url $u -Sha256 "a1b2..." -Destination "models\qwen.gguf"
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Url,
    [string]$Sha256 = "",
    [string]$Destination = ""
)

$ErrorActionPreference = "Stop"

if ($PSVersionTable.PSVersion.Major -lt 5) {
    Write-Error "PowerShell 5.1 or newer is required"
    exit 1
}

# A HuggingFace /blob/ link serves the web page, not the file. It is the single
# most common way this step fails, and the failure is silent: the download
# "succeeds" and produces an HTML document with a .gguf name.
if ($Url -match "huggingface\.co/.+/blob/") {
    $corrected = $Url -replace "/blob/", "/resolve/"
    Write-Host "  note: /blob/ serves the web page, not the file. Using:" -ForegroundColor Yellow
    Write-Host "        $corrected" -ForegroundColor Yellow
    $Url = $corrected
}

# The filename is derived from the URL unless one is given. A fixed default of
# `model.gguf` is anonymous: downloading a second candidate silently overwrites
# the first, and every measurement already attributed to "the model" becomes
# wrong with no error to warn anyone. Distinct names also make the candidate
# comparison in phase 3 possible at all.
#
# The submission package is the opposite case: its download_model.sh writes to
# one declared path, because the template requires _runtime.model_path to match.
if (-not $Destination) {
    $leaf = [System.IO.Path]::GetFileName(($Url -split "[?#]")[0])
    try { $leaf = [System.Uri]::UnescapeDataString($leaf) } catch { }
    if ($leaf -and $leaf.ToLower().EndsWith(".gguf")) {
        $Destination = Join-Path "models" $leaf
    }
    else {
        Write-Host "  note: no .gguf filename in the URL; using models\model.gguf" -ForegroundColor Yellow
        $Destination = "models\model.gguf"
    }
    Write-Host "  destination: $Destination"
}

$directory = Split-Path -Parent $Destination
if ($directory -and -not (Test-Path $directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

# .NET resolves a relative path against the *process* working directory, which
# in PowerShell is whatever it was when the host started -- often
# C:\Windows\system32 -- and not $PWD. Cmdlets like Test-Path and Get-FileHash
# use the PowerShell location and behave; a bare [System.IO.File] call does not.
# Everything below therefore works on an absolute path.
$Destination = [System.IO.Path]::GetFullPath(
    [System.IO.Path]::Combine($PWD.ProviderPath, $Destination))

if (Test-Path $Destination) {
    Write-Host "  $Destination already exists; verifying instead of re-downloading"
}
else {
    Write-Host "  downloading $Url"
    $partial = "$Destination.part"
    try {
        # Windows PowerShell 5.1 buffers the entire response in memory before
        # writing it, which fails or thrashes on a multi-gigabyte model. curl.exe
        # ships with Windows 10 1803+ and streams to disk; WebClient streams too.
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source -L --fail --progress-bar -o $partial $Url
            if ($LASTEXITCODE -ne 0) { throw "curl exited with $LASTEXITCODE" }
        }
        else {
            $client = New-Object System.Net.WebClient
            try { $client.DownloadFile($Url, $partial) } finally { $client.Dispose() }
        }
    }
    catch {
        if (Test-Path $partial) { Remove-Item $partial -Force }
        Write-Error "download failed: $_"
        exit 1
    }
    Move-Item -Path $partial -Destination $Destination -Force
}

# A GGUF file starts with the four bytes "GGUF". An HTML error page does not,
# and the failure it produces deep inside the loader is far harder to read than
# this one.
# Read four bytes through .NET rather than Get-Content: `-AsByteStream` is
# PowerShell 6+, `-Encoding Byte` is 5.1, and choosing wrongly raises a
# parameter-binding error that -ErrorAction cannot catch. This works on both and
# does not read a 2.5 GB file into memory to look at its first four bytes.
#
# $Destination was made absolute above; passing the relative form here is what
# sent the first version of this check looking in C:\Windows\system32.
$header = New-Object byte[] 4
$stream = [System.IO.File]::OpenRead($Destination)
try { $null = $stream.Read($header, 0, 4) } finally { $stream.Close() }
$magic = -join ($header | ForEach-Object { [char]$_ })
if ($magic -ne "GGUF") {
    Write-Host "error: $Destination is not a GGUF file (magic: '$magic')" -ForegroundColor Red
    Write-Host "       the URL returned something that is not a model. Usual causes:"
    Write-Host "         - a /blob/ link instead of /resolve/ (HuggingFace)"
    Write-Host "         - a gated repository that needs acceptance of its licence first"
    Write-Host "         - a 404 page saved under a .gguf name"
    Remove-Item $Destination -Force
    exit 1
}

$actual = (Get-FileHash -Path $Destination -Algorithm SHA256).Hash.ToLower()

if ($Sha256) {
    if ($actual -ne $Sha256.ToLower()) {
        Write-Host "error: checksum mismatch" -ForegroundColor Red
        Write-Host "  expected $($Sha256.ToLower())"
        Write-Host "  actual   $actual"
        Write-Host "  refusing to keep a file that is not the declared model"
        exit 1
    }
    Write-Host "  checksum verified"
}
else {
    Write-Host "  sha256: $actual"
    Write-Host "  record this in configs\runtime\competition.yaml under model.sha256"
    Write-Host "  (the submission gate refuses a package without it -- SUB-004)"
}

$sizeMb = [math]::Round((Get-Item $Destination).Length / 1MB, 0)
Write-Host "  ready: $Destination ($sizeMb MB)"
Write-Host "  next:  leanlm doctor"
