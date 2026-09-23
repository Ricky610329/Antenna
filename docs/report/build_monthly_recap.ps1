param(
    [string]$ManifestPath = (Join-Path $PSScriptRoot 'monthly_recap_2026.json'),
    [string]$BuildDir = 'tmp/monthly-recap/build'
)

# Edit a copy of the current report with native PowerPoint objects.
# Requires desktop PowerPoint. Final delivery is a separate, reviewed copy step.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$source = Join-Path $repoRoot $manifest.source
$build = [IO.Path]::GetFullPath((Join-Path $repoRoot $BuildDir))
if (-not $build.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Build directory must be inside the repository.'
}
$beforeDir = Join-Path $build 'before'
$afterDir = Join-Path $build 'after'
New-Item -ItemType Directory -Force -Path $build, $beforeDir, $afterDir | Out-Null
$draft = Join-Path $build 'monthly-recap.pptx'
$pdf = Join-Path $build 'monthly-recap.pdf'
Copy-Item -LiteralPath $source -Destination $draft -Force

function Color-Value([string]$Hex) {
    $hexValue = $Hex.TrimStart('#')
    return [Convert]::ToInt32($hexValue.Substring(0, 2), 16) +
        256 * [Convert]::ToInt32($hexValue.Substring(2, 2), 16) +
        65536 * [Convert]::ToInt32($hexValue.Substring(4, 2), 16)
}

function Add-Text($Slide, $Item) {
    $shape = $Slide.Shapes.AddTextbox(1, $Item.x, $Item.y, $Item.w, $Item.h)
    $shape.Line.Visible = 0
    $shape.Fill.Visible = 0
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $text = $shape.TextFrame.TextRange
    $text.Text = $Item.text
    $text.Font.Name = 'Microsoft JhengHei'
    $text.Font.NameFarEast = 'Microsoft JhengHei'
    $text.Font.Size = [single]$Item.size
    $text.Font.Bold = $(if ($Item.bold) { -1 } else { 0 })
    $text.Font.Color.RGB = Color-Value $(if ($Item.color) { $Item.color } else { '#152C3D' })
    $text.ParagraphFormat.Alignment = $(if ($Item.align) { $Item.align } else { 1 })
    $text.ParagraphFormat.LineRuleWithin = -1
    $text.ParagraphFormat.SpaceWithin = 1
    $text.ParagraphFormat.SpaceAfter = 0
    $text.ParagraphFormat.SpaceBefore = 0
    $shape.TextFrame2.TextRange.Font.Name = 'Microsoft JhengHei'
    $shape.TextFrame2.TextRange.Font.NameFarEast = 'Microsoft JhengHei'
    $shape.Height = [single]$Item.h
    if ($Item.name) { $shape.Name = $Item.name }
    return $shape
}

function Add-Element($Slide, $Item) {
    switch ($Item.type) {
        'text' { Add-Text $Slide $Item | Out-Null }
        'image' {
            $path = Join-Path $repoRoot $Item.path
            $picture = $Slide.Shapes.AddPicture($path, 0, -1, 0, 0, -1, -1)
            $picture.LockAspectRatio = 0
            $picture.ScaleWidth(1, -1, 0)
            $picture.ScaleHeight(1, -1, 0)
            if ($Item.crop) {
                # Crop the presentation object; keep the original evidence file intact.
                $originalWidth = [double]$picture.Width
                $originalHeight = [double]$picture.Height
                $picture.PictureFormat.CropLeft = [single]($originalWidth * $Item.crop.left)
                $picture.PictureFormat.CropRight = [single]($originalWidth * $Item.crop.right)
                $picture.PictureFormat.CropTop = [single]($originalHeight * $Item.crop.top)
                $picture.PictureFormat.CropBottom = [single]($originalHeight * $Item.crop.bottom)
            }
            $ratio = [double]$picture.Width / [double]$picture.Height
            $width = [Math]::Min([double]$Item.w, [double]$Item.h * $ratio)
            $height = $width / $ratio
            $picture.LockAspectRatio = -1
            $picture.Width = [single]$width
            $picture.Left = [single]($Item.x + ($Item.w - $width) / 2)
            $picture.Top = [single]($Item.y + ($Item.h - $height) / 2)
        }
        'rect' {
            $shape = $Slide.Shapes.AddShape(1, $Item.x, $Item.y, $Item.w, $Item.h)
            $shape.Fill.Solid()
            $shape.Shadow.Visible = 0
            $shape.Fill.ForeColor.RGB = Color-Value $Item.fill
            if ($Item.line) {
                $shape.Line.ForeColor.RGB = Color-Value $Item.line
                $shape.Line.Weight = 1
            } else { $shape.Line.Visible = 0 }
        }
        'line' {
            $shape = $Slide.Shapes.AddLine($Item.x, $Item.y, $Item.x2, $Item.y2)
            $shape.Shadow.Visible = 0
            $shape.Line.ForeColor.RGB = Color-Value $Item.color
            $shape.Line.Weight = [single]$(if ($Item.weight) { $Item.weight } else { 1 })
            if ($Item.arrow) { $shape.Line.EndArrowheadStyle = 3 }
        }
        default { throw "Unknown element type: $($Item.type)" }
    }
}

$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $presentation = $powerPoint.Presentations.Open($draft, 0, 0, 0)
    if ($presentation.PageSetup.SlideWidth -ne 960 -or $presentation.PageSetup.SlideHeight -ne 540) {
        throw 'Unexpected source slide dimensions.'
    }
    $baseCount = [int]$manifest.preserve_slides
    if ($presentation.Slides.Count -le $baseCount) { throw 'Expected the existing recap after the main report.' }
    for ($index = $presentation.Slides.Count; $index -gt $baseCount; $index--) {
        $slide = $presentation.Slides.Item($index)
        $hasRecapTitle = $false
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                if ($shape.Name -eq 'recap_title' -or $shape.TextFrame.TextRange.Text -eq $manifest.old_recap_title) {
                    $hasRecapTitle = $true
                }
            }
        }
        if (-not $hasRecapTitle) { throw "Refusing to replace an unrecognized slide: $index" }
        $slide.Delete()
    }
    for ($index = 1; $index -le $baseCount; $index++) {
        $presentation.Slides.Item($index).Export((Join-Path $beforeDir ('slide-{0:D2}.png' -f $index)), 'PNG', 1600, 900)
    }
    foreach ($definition in $manifest.slides) {
        $slide = $presentation.Slides.Add($presentation.Slides.Count + 1, 12)
        Add-Text $slide ([pscustomobject]@{
            x=42; y=22; w=875; h=46; text=$definition.title; size=30; bold=$true; name='recap_title'
        }) | Out-Null
        Add-Element $slide ([pscustomobject]@{type='line'; x=42; y=82; x2=918; y2=82; color='#C6D5E1'; weight=0.7})
        Add-Element $slide ([pscustomobject]@{type='line'; x=42; y=82; x2=140; y2=82; color='#245B85'; weight=2.3})
        foreach ($item in $definition.elements) { Add-Element $slide $item }
        Add-Text $slide ([pscustomobject]@{
            x=42; y=504; w=830; h=19; text=$definition.source_label; size=9; color='#526A7D'
        }) | Out-Null
        Add-Text $slide ([pscustomobject]@{
            x=893; y=507; w=25; h=18; text=[string]$slide.SlideIndex; size=11; color='#526A7D'; align=3
        }) | Out-Null
        foreach ($shape in $slide.NotesPage.Shapes) {
            if ($shape.Type -eq 14 -and $shape.PlaceholderFormat.Type -eq 2) {
                $shape.TextFrame.TextRange.Text = $definition.notes
                break
            }
        }
    }
    $expected = $baseCount + $manifest.slides.Count
    if ($presentation.Slides.Count -ne $expected) { throw 'Unexpected final slide count.' }
    $presentation.Save()
    $presentation.SaveAs($pdf, 32)
    $overflow = @()
    foreach ($slide in $presentation.Slides) {
        $slide.Export((Join-Path $afterDir ('slide-{0:D2}.png' -f $slide.SlideIndex)), 'PNG', 1600, 900)
        if ($slide.SlideIndex -gt $baseCount) {
            foreach ($shape in $slide.Shapes) {
                if ($shape.HasTextFrame -and $shape.TextFrame.HasText -and $shape.TextFrame.TextRange.BoundHeight -gt $shape.Height + 2) {
                    $overflow += "slide-$($slide.SlideIndex):height=$($shape.Height), bound=$($shape.TextFrame.TextRange.BoundHeight):$($shape.TextFrame.TextRange.Text)"
                }
            }
        }
    }
    for ($index = 1; $index -le $baseCount; $index++) {
        $name = 'slide-{0:D2}.png' -f $index
        if ((Get-FileHash -LiteralPath (Join-Path $beforeDir $name)).Hash -ne (Get-FileHash -LiteralPath (Join-Path $afterDir $name)).Hash) {
            throw "Original slide changed: $index"
        }
    }
    [pscustomobject]@{
        slide_count=$expected; original_slides_unchanged=$true; overflow=$overflow; pptx=$draft; pdf=$pdf
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
    if ($overflow.Count -gt 0) { throw ('Review text overflow: ' + ($overflow -join '; ')) }
    Write-Output "Built $expected slides; original $baseCount renders unchanged."
    Write-Output $draft
    Write-Output $pdf
}
finally {
    if ($presentation -ne $null) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    if ($powerPoint -ne $null) {
        if ($powerPoint.Presentations.Count -eq 0) { $powerPoint.Quit() }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
