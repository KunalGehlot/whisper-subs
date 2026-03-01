# Whisper Subs

Generate bilingual subtitles (German + English) and an analysis report from German-language videos using OpenAI's Whisper and GPT-4o APIs.

## Features

- Extracts audio from video files (MP4, MOV, MKV, AVI, WebM)
- Automatically splits large audio to stay within Whisper's 25 MB upload limit
- Transcribes German speech to text using [Whisper](https://platform.openai.com/docs/guides/speech-to-text)
- Generates German `.de.srt` subtitle files
- Translates subtitles to English and generates `.en.srt` files using GPT-4o
- Produces a structured analysis report (topic, key points, action items, notable quotes)

## Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) installed and available on your PATH
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

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt-get install ffmpeg

# Windows
choco install ffmpeg
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

## License

[MIT](LICENSE)
