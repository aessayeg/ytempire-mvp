"""
FFmpeg Installation Helper for Windows
Downloads and sets up FFmpeg for video processing
"""
import os
import sys
import zipfile
import requests
import shutil
from pathlib import Path


def download_ffmpeg():
    """Download FFmpeg for Windows"""
    print("=" * 60)
    print("FFmpeg Installation Helper")
    print("=" * 60)

    # FFmpeg download URL (Windows 64-bit)
    ffmpeg_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"

    # Create ffmpeg directory in project
    ffmpeg_dir = Path("ffmpeg")
    ffmpeg_dir.mkdir(exist_ok=True)

    zip_path = ffmpeg_dir / "ffmpeg.zip"

    print(f"\n1. Downloading FFmpeg from GitHub...")
    print(f"   URL: {ffmpeg_url}")

    try:
        # Download with progress
        response = requests.get(ffmpeg_url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with open(zip_path, "wb") as f:
            downloaded = 0
            for data in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                downloaded += len(data)
                f.write(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(
                        f"   Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)",
                        end="\r",
                    )

        print(f"\n   [OK] Downloaded to {zip_path}")

    except Exception as e:
        print(f"   [ERROR] Failed to download: {e}")
        return False

    print(f"\n2. Extracting FFmpeg...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ffmpeg_dir)
        print(f"   [OK] Extracted to {ffmpeg_dir}")

        # Find the extracted folder
        extracted_folders = [
            f for f in ffmpeg_dir.iterdir() if f.is_dir() and "ffmpeg" in f.name.lower()
        ]
        if extracted_folders:
            extracted_folder = extracted_folders[0]
            bin_folder = extracted_folder / "bin"

            if bin_folder.exists():
                # Move executables to project ffmpeg folder
                for exe in bin_folder.glob("*.exe"):
                    dest = ffmpeg_dir / exe.name
                    shutil.move(str(exe), str(dest))
                    print(f"   [OK] Moved {exe.name} to {ffmpeg_dir}")

                # Clean up extracted folder
                shutil.rmtree(extracted_folder)

                # Remove zip file
                zip_path.unlink()

                print(f"\n3. FFmpeg Installation Complete!")
                print(f"   Location: {ffmpeg_dir.absolute()}")
                print(f"   Executables: ffmpeg.exe, ffprobe.exe, ffplay.exe")

                # Test ffmpeg
                ffmpeg_exe = ffmpeg_dir / "ffmpeg.exe"
                if ffmpeg_exe.exists():
                    print(f"\n4. Testing FFmpeg...")
                    import subprocess

                    try:
                        result = subprocess.run(
                            [str(ffmpeg_exe), "-version"],
                            capture_output=True,
                            text=True,
                            timeout=5,
                        )
                        if result.returncode == 0:
                            version_line = result.stdout.split("\n")[0]
                            print(f"   [OK] {version_line}")
                        else:
                            print(f"   [WARNING] FFmpeg test returned non-zero code")
                    except Exception as e:
                        print(f"   [ERROR] Failed to test ffmpeg: {e}")

                return True
            else:
                print(f"   [ERROR] bin folder not found in extracted content")
                return False

    except Exception as e:
        print(f"   [ERROR] Failed to extract: {e}")
        return False


def update_path_in_video_processor():
    """Update VideoProcessor to use local FFmpeg"""
    print(f"\n5. Updating VideoProcessor to use local FFmpeg...")

    video_processor_path = Path("app/services/video_processor.py")
    if video_processor_path.exists():
        with open(video_processor_path, "r") as f:
            content = f.read()

        # Check if already updated
        if "ffmpeg/ffmpeg.exe" in content:
            print("   [OK] VideoProcessor already configured")
        else:
            # Update the _find_ffmpeg method to check local folder first
            old_paths = """paths = [
            "ffmpeg",  # System PATH
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe",
            "C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe"
        ]"""

            new_paths = """paths = [
            str(Path(__file__).parent.parent.parent / "ffmpeg" / "ffmpeg.exe"),  # Local project ffmpeg
            "ffmpeg",  # System PATH
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "C:\\\\ffmpeg\\\\bin\\\\ffmpeg.exe",
            "C:\\\\Program Files\\\\ffmpeg\\\\bin\\\\ffmpeg.exe"
        ]"""

            if old_paths in content:
                content = content.replace(old_paths, new_paths)
                with open(video_processor_path, "w") as f:
                    f.write(content)
                print("   [OK] Updated VideoProcessor to use local FFmpeg")
            else:
                print(
                    "   [INFO] VideoProcessor code structure different, manual update may be needed"
                )
    else:
        print("   [WARNING] VideoProcessor file not found")


def main():
    """Main installation process"""
    success = download_ffmpeg()

    if success:
        update_path_in_video_processor()

        print("\n" + "=" * 60)
        print("INSTALLATION COMPLETE!")
        print("=" * 60)
        print("\nFFmpeg has been installed successfully.")
        print("The video processing pipeline should now work properly.")
        print("\nYou can now:")
        print("1. Run video generation tests")
        print("2. Process videos with FFmpeg")
        print("3. Generate complete videos with audio and visuals")
    else:
        print("\n" + "=" * 60)
        print("INSTALLATION FAILED")
        print("=" * 60)
        print("\nPlease try the following:")
        print("1. Check your internet connection")
        print("2. Try running this script again")
        print("3. Manually download FFmpeg from https://ffmpeg.org/download.html")
        print("   and place ffmpeg.exe in the 'ffmpeg' folder")


if __name__ == "__main__":
    main()
