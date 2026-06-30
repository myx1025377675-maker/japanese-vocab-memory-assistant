Add-Type -AssemblyName System.Drawing

$OutDir = Join-Path $PSScriptRoot "output"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$FontRegular = "C:\Windows\Fonts\msyh.ttc"
$FontBold = "C:\Windows\Fonts\msyhbd.ttc"
$FontJp = "C:\Windows\Fonts\YuGothM.ttc"

function New-Font($path, $size, $style = [System.Drawing.FontStyle]::Regular) {
    $private = New-Object System.Drawing.Text.PrivateFontCollection
    $private.AddFontFile($path)
    return New-Object System.Drawing.Font($private.Families[0], $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Brush($hex) {
    return New-Object System.Drawing.SolidBrush([System.Drawing.ColorTranslator]::FromHtml($hex))
}

function Pen($hex, $width = 2) {
    return New-Object System.Drawing.Pen([System.Drawing.ColorTranslator]::FromHtml($hex), $width)
}

function Measure-Line($g, $text, $font) {
    return $g.MeasureString($text, $font, 20000, [System.Drawing.StringFormat]::GenericTypographic).Width
}

function Wrap-Text($g, $text, $font, $maxWidth) {
    $lines = New-Object System.Collections.Generic.List[string]
    foreach ($raw in ($text -split "`n")) {
        if ($raw.Trim().Length -eq 0) {
            $lines.Add("")
            continue
        }
        $line = ""
        foreach ($ch in $raw.ToCharArray()) {
            $candidate = "$line$ch"
            if ((Measure-Line $g $candidate $font) -le $maxWidth -or $line.Length -eq 0) {
                $line = $candidate
            } else {
                $lines.Add($line)
                $line = "$ch"
            }
        }
        if ($line.Length -gt 0) { $lines.Add($line) }
    }
    return $lines
}

function Fill-RoundedRect($g, $brush, $x, $y, $w, $h, $r) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($x, $y, $r, $r, 180, 90)
    $path.AddArc($x + $w - $r, $y, $r, $r, 270, 90)
    $path.AddArc($x + $w - $r, $y + $h - $r, $r, $r, 0, 90)
    $path.AddArc($x, $y + $h - $r, $r, $r, 90, 90)
    $path.CloseFigure()
    $g.FillPath($brush, $path)
    $path.Dispose()
}

function Stroke-RoundedRect($g, $pen, $x, $y, $w, $h, $r) {
    $path = New-Object System.Drawing.Drawing2D.GraphicsPath
    $path.AddArc($x, $y, $r, $r, 180, 90)
    $path.AddArc($x + $w - $r, $y, $r, $r, 270, 90)
    $path.AddArc($x + $w - $r, $y + $h - $r, $r, $r, 0, 90)
    $path.AddArc($x, $y + $h - $r, $r, $r, 90, 90)
    $path.CloseFigure()
    $g.DrawPath($pen, $path)
    $path.Dispose()
}

function Draw-TextBlock($g, $text, $font, $brush, $x, $y, $maxWidth, $lineHeight) {
    $lines = Wrap-Text $g $text $font $maxWidth
    foreach ($line in $lines) {
        $g.DrawString($line, $font, $brush, $x, $y)
        $y += $lineHeight
    }
    return $y
}

function Draw-Pill($g, $text, $x, $y, $font, $fill, $fg) {
    $width = [int](Measure-Line $g $text $font) + 46
    Fill-RoundedRect $g (Brush $fill) $x $y $width 48 24
    $g.DrawString($text, $font, (Brush $fg), ($x + 23), ($y + 10))
    return $width
}

function Draw-Header($g, $num, $title, $subtitle, $accent) {
    Draw-Pill $g "中日音读速记 / card $num" 104 88 (New-Font $FontBold 28) $accent "#ffffff" | Out-Null
    $g.DrawString($title, (New-Font $FontBold 76), (Brush "#2f2a27"), 104, 158)
    $g.DrawString($subtitle, (New-Font $FontRegular 31), (Brush "#766c66"), 108, 250)
    $g.DrawLine((Pen $accent 5), 108, 322, 1492, 322)
}

function Draw-Section($g, $x, [ref]$y, $title, $body, $accent, $fill = "#fffaf3") {
    $top = $y.Value
    $innerX = $x + 34
    $maxWidth = 1330
    $titleFont = New-Font $FontBold 38
    $bodyFont = New-Font $FontRegular 29
    $titleHeight = 56
    $lines = Wrap-Text $g $body $bodyFont $maxWidth
    $height = 46 + $titleHeight + ($lines.Count * 43) + 34
    Fill-RoundedRect $g (Brush $fill) $x $top 1392 $height 36
    Stroke-RoundedRect $g (Pen "#eadfd3" 2) $x $top 1392 $height 36
    Fill-RoundedRect $g (Brush $accent) $innerX ($top + 34) 16 40 8
    $g.DrawString($title, $titleFont, (Brush "#2f2a27"), ($innerX + 32), ($top + 24))
    $bodyY = $top + 92
    foreach ($line in $lines) {
        $g.DrawString($line, $bodyFont, (Brush "#4b4642"), $innerX, $bodyY)
        $bodyY += 43
    }
    $y.Value = $top + $height + 28
}

function Draw-Card($filename, $num, $title, $subtitle, $accent, $sections) {
    $bmp = New-Object System.Drawing.Bitmap(1600, 2200)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.ColorTranslator]::FromHtml("#f7efe6"))

    for ($i = 0; $i -lt 1600; $i += 80) {
        $g.DrawLine((Pen "#efe3d7" 1), $i, 0, $i, 2200)
    }
    for ($j = 0; $j -lt 2200; $j += 80) {
        $g.DrawLine((Pen "#efe3d7" 1), 0, $j, 1600, $j)
    }

    Fill-RoundedRect $g (Brush "#fffdf8") 64 54 1472 2088 50
    Stroke-RoundedRect $g (Pen "#e2d5c8" 3) 64 54 1472 2088 50
    Draw-Header $g $num $title $subtitle $accent

    $y = 372
    foreach ($s in $sections) {
        Draw-Section $g 104 ([ref]$y) $s.Title $s.Body $accent $s.Fill
    }

    $footerFont = New-Font $FontRegular 24
    $g.DrawString("严谨用法：这些都是「常见对应/高概率线索」，不要当成一一换算公式。", $footerFont, (Brush "#8b8178"), 108, 2070)
    $g.DrawString("2026 · Japanese Onyomi Notes", $footerFont, (Brush "#b0a59b"), 108, 2110)

    $path = Join-Path $OutDir $filename
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose()
    $bmp.Dispose()
    Write-Output $path
}

$cards = @(
    @{
        File="01_overview.png"; Num="01"; Title="中日音读对应总览"; Subtitle="用中文拼音缩小范围，用日语规则校准读音"; Accent="#d98973";
        Sections=@(
            @{Title="总原则"; Fill="#fffaf3"; Body="现代普通话拼音不能直接换算日语音读。它只是线索：两者都和中古汉语有关，但各自经历了不同演变。"},
            @{Title="最可靠的三条"; Fill="#f7fbf7"; Body="1. 普通话 -n 韵尾常对应日语拨音 ん：安→あん，心→しん，天→てん。`n2. 普通话 -ng 韵尾常对应长音或拗长音：東→とう，中→ちゅう，英→えい，京→きょう。`n3. 日语音读以 く・つ・ち・き 收尾，多对应古入声字；现代普通话通常没有 n/ng 尾。"},
            @{Title="使用方法"; Fill="#f7f7ff"; Body="先看韵母判断「长音 / ん / 入声尾」，再看声母锁定假名行，最后用促音化、半浊音化等日语内部规则修正。"},
            @{Title="警惕词"; Fill="#fff6f3"; Body="不要使用「精准、必须、绝对、100%」这类表述。更严谨的说法是：常见、倾向、大概率、高度稳定。"}
        )
    },
    @{
        File="02_final_vowels.png"; Num="02"; Title="韵母：元音与尾音"; Subtitle="判断长音、拨音和特殊结尾"; Accent="#7d9f8c";
        Sections=@(
            @{Title="-n → ん"; Fill="#f7fbf7"; Body="普通话 an / en / in / un / ian / uan 常对应拨音 ん。`n例：安 an→あん，心 xin→しん，天 tian→てん，根 gen→こん，本 ben→ほん，門 men→もん。"},
            @{Title="-ng → 长音"; Fill="#fffaf3"; Body="普通话 ang / eng / ing / ong 常对应日语长音，如 おう、えい、ょう。`n例：東 dong→とう，中 zhong→ちゅう，王 wang→おう，英 ying→えい，京 jing→きょう，能 neng→のう。"},
            @{Title="ing 不只等于 えい"; Fill="#f7f7ff"; Body="ing 常见对应包括 えい 和拗长音 ょう。`n例：英→えい，令→れい，定→てい；京→きょう，清→せい / しょう，明→めい / みょう。"},
            @{Title="ao / iao 与 ui"; Fill="#fff6f3"; Body="ao / iao 常对应 おう、ょう 等长音：高→こう，老→ろう，表→ひょう，張→ちょう，校→こう。`nui / uei 常对应 い 结尾：水→すい，会→かい，悔→かい，対/對→たい。"}
        )
    },
    @{
        File="03_initials.png"; Num="03"; Title="声母：锁定假名行"; Subtitle="辅音只能缩小范围，不能直接推出唯一读音"; Accent="#9b8bc6";
        Sections=@(
            @{Title="高频对应"; Fill="#f7f7ff"; Body="b / p / f → は・ば行：半→はん，法→ほう，白→はく / びゃく。`ng / k / h → か・が行：高→こう，海→かい，国→こく。`nd / t → た・だ行：大→だい，天→てん，定→てい。"},
            @{Title="さ行与分散区"; Fill="#fffaf3"; Body="z / c / s / sh → さ・ざ行较常见：三→さん，思→し，書→しょ。`nzh / ch 常见到 た行、さ行等：中→ちゅう，張→ちょう，知→ち。`nj / q / x 分散更大：心→しん，学→がく，京→きょう，気/氣→き。"},
            @{Title="l / m / n"; Fill="#f7fbf7"; Body="l → ら行高度稳定：理→り，留→りゅう，来→らい，礼→れい。`nm → ま行或ば行：満→まん，美→び，馬→ば / め。`nn → な行或だ行：内→ない，南→なん，男→だん，年→ねん。"},
            @{Title="为什么会有例外"; Fill="#fff6f3"; Body="日语汉字音有吴音、汉音、唐音、惯用音等层次。同一个字可能多个音读，所以声母规律适合「缩小搜索范围」，不适合「直接背答案」。"}
        )
    },
    @{
        File="04_sound_changes.png"; Num="04"; Title="日语内部音变"; Subtitle="组合成词后，还要看促音、半浊音和连浊"; Accent="#c28b5e";
        Sections=@(
            @{Title="促音化：く / つ + かさたは"; Fill="#fffaf3"; Body="汉字音以 く・つ 结尾，后接 か・さ・た・は 行时，常变促音 っ。`n例：学校 がく+こう→がっこう，国会 こく+かい→こっかい，物件 ぶつ+けん→ぶっけん。"},
            @{Title="接 は行常变 ぱ行"; Fill="#fff6f3"; Body="つ / く 促音化后接 は行，常伴随半浊音化。`n例：発表 はつ+ひょう→はっぴょう，失敗 しつ+はい→しっぱい。"},
            @{Title="ん + は行"; Fill="#f7fbf7"; Body="在许多汉语音读熟词中，ん 后接 は行常变 ぱ行。`n例：鉛筆 えん+ひつ→えんぴつ，先輩 せん+はい→せんぱい，散歩 さん+ほ→さんぽ。`n注意：这不是无条件机械规则。"},
            @{Title="连浊：主要是和语复合"; Fill="#f7f7ff"; Body="两个独立词复合时，后项か・さ・た・は行可能浊化。`n例：手紙 て+かみ→てがみ，鼻血 はな+ち→はなぢ，山桜 やま+さくら→やまざくら。`n汉语音读词通常不能靠连浊预测，如 会社→かいしゃ。"},
            @{Title="反推：入声字"; Fill="#fffaf3"; Body="日语音读以 く・つ・ち・き 收尾，多为古入声字；现代普通话通常没有 -n / -ng。`n例：国→こく，学→がく，鉄→てつ，日→にち，白→はく，石→せき，仏→ぶつ，別→べつ。"}
        )
    }
)

foreach ($card in $cards) {
    Draw-Card $card.File $card.Num $card.Title $card.Subtitle $card.Accent $card.Sections
}
