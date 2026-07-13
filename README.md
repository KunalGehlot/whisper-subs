# Whisper Subs

Generate bilingual subtitles (German + English) and an analysis report from German-language videos using OpenAI's Whisper and GPT-4o APIs.

## Features

- Extracts audio from video files (MP4, MOV, MKV, AVI, WebM)
- Splits audio at long pauses (silence detection) so it stays within Whisper's
  25 MB upload limit **and** never asks Whisper to transcribe across a long
  silence — the situation that otherwise corrupts its timestamps
- Anchors every chunk to its real start time, so subtitle timing stays accurate
  no matter how long the speaker pauses
- Transcribes German speech to text using [Whisper](https://platform.openai.com/docs/guides/speech-to-text)
- Transcribes chunks in parallel and tolerates a failed chunk without aborting the run
- Generates German `.de.srt` subtitle files
- Translates subtitles to English and generates `.en.srt` files using GPT-4o
- Produces a structured analysis report (topic, key points, action items, notable quotes)
- Filters out common Whisper hallucinations (phantom subtitles, attribution text)
- Supports bundled ffmpeg binaries for systems without ffmpeg in PATH

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) (see [Installing ffmpeg](#installing-ffmpeg) below)
- An [OpenAI API key](https://platform.openai.com/api-keys)

## Quick Start

```bash
# Clone the repo
git clone https://github.com/KunalGehlot/whisper-subs.git
cd whisper-subs

# Copy the example env file and add your API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# Run (creates venv, installs deps, and processes the video)
./run.sh path/to/video.mp4
```

Or set up manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python process_video.py path/to/video.mp4
```

## Installing ffmpeg

### macOS

```bash
brew install ffmpeg
```

### Ubuntu / Debian

```bash
sudo apt-get install ffmpeg
```

### Windows

**Option A** - Install via package manager:

```bash
choco install ffmpeg
```

**Option B** - Bundled binaries (no install required):

Download ffmpeg from [ffmpeg.org/download](https://ffmpeg.org/download.html) and extract it into the project directory. The tool automatically searches for `ffmpeg*/bin/` folders in the project root before falling back to the system PATH.

```
whisper-subs/
  ffmpeg-8.0.1/
    bin/
      ffmpeg.exe
      ffprobe.exe
  process_video.py
  ...
```

## Output

For a file called `lecture.mp4`, the tool generates:

| File | Description |
|---|---|
| `lecture.de.srt` | German subtitles |
| `lecture.en.srt` | English subtitles |
| `lecture_report.md` | Analysis report |

SRT files are placed next to the source video so media players like VLC auto-detect them.

## Usage

```
usage: process_video.py [-h] [--api-key API_KEY] video

Generate German/English subtitles and a key-points report from a German-language MP4 video.

positional arguments:
  video              Path to the MP4 video file

options:
  -h, --help         show this help message and exit
  --api-key API_KEY  OpenAI API key (overrides OPENAI_API_KEY env var)
```

## Running Tests

```bash
# Offline tests (no API key needed)
python -m unittest discover tests -v

# Full test suite including live transcription
OPENAI_API_KEY=sk-... python -m unittest discover tests -v
```

## License

[MIT](LICENSE)
