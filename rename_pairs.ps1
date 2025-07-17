# Windows PowerShell Script to Rename Training Pairs
# Run this in PowerShell from your project directory

# Navigate to training pairs directory
cd data/training_pairs

# Count existing properly named files to know where to start
$existingCount = (Get-ChildItem "iphone_*.dng" | Measure-Object).Count
$startNum = $existingCount + 1

Write-Host "Found $existingCount existing pairs"
Write-Host "Starting new numbering from $startNum"

# Rename iPhone files (IMG_*.DNG -> iphone_XXX.dng)
$counter = $startNum
Get-ChildItem "IMG_*.DNG" | Sort-Object Name | ForEach-Object {
    $newName = "iphone_{0:D3}.dng" -f $counter
    Rename-Item $_.Name $newName
    Write-Host "Renamed $($_.Name) -> $newName"
    $counter++
}

# Rename Sony files (DSC*.ARW -> sony_XXX.arw)  
$counter = $startNum
Get-ChildItem "DSC*.ARW" | Sort-Object Name | ForEach-Object {
    $newName = "sony_{0:D3}.arw" -f $counter
    Rename-Item $_.Name $newName
    Write-Host "Renamed $($_.Name) -> $newName"
    $counter++
}

# Verify results
$iPhoneCount = (Get-ChildItem "iphone_*.dng" | Measure-Object).Count
$sonyCount = (Get-ChildItem "sony_*.arw" | Measure-Object).Count

Write-Host ""
Write-Host "RENAMING COMPLETE!"
Write-Host "iPhone files: $iPhoneCount"
Write-Host "Sony files: $sonyCount"
Write-Host "Total pairs: $iPhoneCount"

if ($iPhoneCount -eq $sonyCount) {
    Write-Host "SUCCESS: All pairs match!"
} else {
    Write-Host "WARNING: Mismatch in file counts!"
}
