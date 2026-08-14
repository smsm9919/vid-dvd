import shutil, subprocess
from pathlib import Path

class MediaError(RuntimeError):
    pass

def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def run(cmd):
    p=subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise MediaError(p.stderr[-5000:])
    return p

def concat_videos(paths, output):
    paths=[Path(p) for p in paths]
    if not paths: raise MediaError("No clips supplied.")
    if len(paths)==1:
        shutil.copy2(paths[0], output)
        return output
    listing=Path(output).with_suffix(".txt")
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in paths), encoding="utf-8")
    run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(listing),"-c","copy",str(output)])
    listing.unlink(missing_ok=True)
    return output

def normalize_vertical(input_path, output_path):
    vf="scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    run(["ffmpeg","-y","-i",str(input_path),"-vf",vf,"-c:v","libx264","-preset","medium","-crf","20","-pix_fmt","yuv420p","-c:a","aac","-b:a","192k","-movflags","+faststart",str(output_path)])
    return output_path
